"""
This module is used to create and run Evaluators for different models.

Evaluators allow for rapid model simulating or solving with input variable manipulation
and output variable filtering.

Currently there are four specific evaluators: EvaluatorEP (for EnergyPlus), `EvaluatorEH` (for PyEHub)
`EvaluatorGeneric` (for custom functions), and `AdaptiveSR` (for adaptive sampling).
The Evaluators wrap their respective modeling tools with the evaluator interface.
"""

# Python Core Libraries
import copy
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Iterable, Sequence
from warnings import warn

# External Libraries
import numpy as np
import pandas as pd
import platypus
import yaml
from dask.dataframe import from_pandas
from deprecated.sphinx import deprecated
from pandas import DataFrame as DF
from pyehub.outputter import output_excel
from tqdm.auto import tqdm

# BESOS Imports
from besos import config
from besos import eplus_funcs as eplus
from besos import eppy_funcs as ef
from besos import pyehub_funcs as pf
from besos import sampling
from besos.problem import Problem
from besos.besostypes import PathLike


def _freeze(value):
    """This function converts values to hashable representations, so that
    they can be used as inputs for caching.

    :param value: a value to attempt to convert to a hashable equivalent
    :return: the value, or a tuple with the same contents.
    """
    # We may want to add other types to this function
    # set -> frozenset, or saving the type of the object
    # in addition to the values.
    # currently these are not needed.

    # note that this try/except is not the same as
    # isinstance(value, Hashable), which
    # incorrectly detects pd.Series as hashable
    try:
        hash(value)
        return value
    except TypeError:
        pass
    if isinstance(value, dict):
        return frozenset((_freeze(k), _freeze(v)) for k, v in value.items())
    if isinstance(value, Iterable):
        return tuple(_freeze(x) for x in value)
    raise TypeError(f"unhashable type: '{type(value)}'")


class AbstractEvaluator(ABC):
    """
    Base class for Evaluators. This template requires that Evaluators are callable.
    It also gives them the df_apply method and result caching.

    Takes a list of values parameterising a model and return objective/constraint results.

    Evaluates each row in a DataFrame and return the output in another DataFrame
    (optionally including the input) and caches results
    """

    error_mode_options = {"Failfast", "Silent", "Print"}

    def __init__(
        self,
        problem: Problem,
        error_mode: str = "Failfast",
        error_value: tuple = None,
        progress_bar: bool = True,
    ):
        """

        :param problem: description of the inputs and outputs the evaluator will use
        :param error_mode: One of {'Failfast', 'Silent', 'Print'}.
            Failfast: Stop evaluating as soon as an error is encountered.
            Silent: Evaluation will return the `error_value` for any input values that raise an error.
            Print: Same as silent, but warnings are printed to stderr for any errors.
        :param error_value: The value of the evaluation if an error occurs.
            Incompatible with error_mode='Failfast'. Must be a tuple consisting of
            values for the objectives followed by values for the constraints.
        :param progress_bar: whether or not to display a progress bar
        """
        self.problem = problem
        self.error_mode = error_mode
        self.error_value = self._convert_error_mode_from_old_format(error_value)
        self.validate_error_mode()
        self._cache = {}
        self.progress_bar = progress_bar
        self.pbar = None

    def validate_error_mode(self) -> None:
        if self.error_mode not in self.error_mode_options:
            raise ValueError(
                f"Invalid error mode '{self.error_mode}', "
                f"only {self.error_mode_options} are allowed"
            )
        if self.error_mode == "Failfast":
            if self.error_value is None:
                return
            else:
                raise ValueError("error value cannot be set when in Failfast mode")
        # intuit error_value if needed
        if self.error_value is None:
            self.error_value = [
                None,
            ] * (self.problem.num_outputs + self.problem.num_constraints)
        if (
            len(self.error_value)
            != self.problem.num_outputs + self.problem.num_constraints
        ):
            raise ValueError("error value must match outputs and constraints length")
        err_out = list(self.error_value[: self.problem.num_outputs])
        err_constraint = list(self.error_value[self.problem.num_outputs :])

        def default(minimize, value):
            if value is None:
                return float("inf") if minimize else float("-inf")
            else:
                return value

        err_out = [
            default(minimize, value)
            for minimize, value in zip(self.problem.minimize_outputs, err_out)
        ]
        for value in err_constraint:
            if value is None:
                raise NotImplementedError(
                    "Cannot Intuit Constraint Error Values."
                    "If you want this feature, request it at"
                    "https://gitlab.com/energyincities/besos/-/issues"
                )
        self.error_value = err_out + err_constraint

    @abstractmethod
    def eval_single(self, values: Sequence, **kwargs) -> tuple:
        """Returns the objective results for a single list of parameter values.

        :param values: A list of values to set each parameter to,
            in the same order as this evaluator's inputs
        :param kwargs: Any keyword arguments
        :return: a tuple of the objectives and constraints
        """
        pass

    def __call__(self, values: Sequence, **kwargs) -> tuple:
        """Returns the objective results for a single list of parameter values.

        :param values: A list of values to set each parameter to,
            in the same order as this evaluator's inputs
        :return: a tuple of the objectives' results
        """
        # Enables validation and caching in subclasses
        # Redirects calls to evaluate a list of values to the eval_single function.
        # Override eval_single, not this method.
        # Values can be empty to allow evaluating at the current state, but this is not the default behaviour
        # If this is not supported, the validate function of a subclass should reject an empty list

        key = _freeze(values), _freeze(kwargs)
        if key in self._cache:
            return self._cache[key]
        try:
            self.validate(values)
            result = self.eval_single(values, **kwargs)
        except Exception as e:
            if self.error_mode != "Silent":
                msg = ""
                if self.problem.num_inputs > 0:
                    msg += f'for inputs: {self.problem.names("inputs")} '
                msg += f"problematic values were: {values}"
                warn(msg)
            if self.error_mode == "Failfast":
                raise e
            else:
                result = self.error_value
        self._cache[key] = result
        return result

    def cache_clear(self) -> None:
        """Clears any cached vales of calls to this evaluator.
        This should be called whenever the evaluator's outputs could have changed."""
        self._cache = {}

    def validate(self, values: Sequence) -> None:
        """Takes a list of values and checks that they are a valid input for this evaluator."""
        if len(values) != self.problem.num_inputs:
            raise ValueError(
                f"Wrong number of input values."
                f"{len(values)} provided, {self.problem.num_inputs} expected"
            )
        for descriptor, value in zip(self.problem.value_descriptors, values):
            if not descriptor.validate(value):
                raise ValueError(f"Invalid value {value} for descriptor {descriptor}")

    def estimate_time(self, df: DF, processes: int = 1) -> None:
        """Prints out a very rough estimate of the amount of time a job will take to complete.
        Will underestimate smaller sample sets but becomes more accurate as they become larger.

        :param df: a DataFrame where each row represents valid input values for this Evaluator.
        :param processes: amount of cores to use
        """
        start = time.time()
        res = self(df.iloc[0])
        end = time.time()

        if processes > 1:
            estimate = (end - start) * df.shape[0] / processes
        else:
            estimate = (end - start) * df.shape[0]
        h = np.floor(estimate / 3600)
        m = np.floor((estimate % 3600) / 60)
        s = np.floor((estimate % 3600) % 60)
        print(f"Estimated time for completion: {h} hours, {m} minutes, {s} seconds")

    def df_apply(self, df: DF, keep_input=False, processes: int = 1, **kwargs) -> DF:
        """Applies this evaluator to an entire dataFrame, row by row.

        :param df: a DataFrame where each row represents valid input values for this Evaluator.
        :param keep_input: whether to include the input data in the returned DataFrame
        :param processes: amount of cores to use
        :return: Returns a DataFrame with one column containing the results for each objective.
        """
        if processes > 1:
            ddf = from_pandas(df, npartitions=processes)
            # TODO: Use information from problem to give accurate types for this shape.
            try:
                if kwargs["keep_dirs"]:
                    expected_output_shape = pd.DataFrame(
                        columns=list(range(self.problem.num_outputs + 1))
                    )
                else:
                    expected_output_shape = pd.DataFrame(
                        columns=list(range(self.problem.num_outputs))
                    )
            except:
                expected_output_shape = pd.DataFrame(
                    columns=list(range(self.problem.num_outputs))
                )
            dask_result = ddf.apply(
                self,
                axis=1,
                meta=expected_output_shape,
                result_type="expand",
                **kwargs,
            )
            result = dask_result.compute()
        else:
            if self.progress_bar:
                self.pbar = tqdm(total=len(df.index), desc="Executing", unit="row")
            result = df.apply(self, axis=1, result_type="expand", **kwargs)
        result = result.rename(
            columns={i: name for i, name in enumerate(self.problem.names("outputs"))},
            errors="raise",
        )
        if keep_input:
            result = pd.concat([df, result], axis=1)
        self.cleanup_pbar()
        return result

    def to_platypus(self) -> platypus.Problem:
        """Converts this evaluator (and the underlying problem) to a platypus compatible format

        :return: A platypus Problem that can optimise over this evaluator
        """
        problem = self.problem.to_platypus()
        problem.function = lambda all_outputs: self.package_for_platypus(
            self(all_outputs)
        )
        return problem

    def package_for_platypus(
        self, all_outputs: tuple
    ) -> Union[Tuple, Tuple[Tuple, Tuple]]:
        if self.problem.num_constraints > 0:
            return (
                all_outputs[: self.problem.num_outputs],
                all_outputs[self.problem.num_outputs :],
            )
        else:
            return all_outputs

    def sample(self, num_samples: int) -> DF:
        inputs = sampling.dist_sampler(sampling.lhs, self.problem, num_samples)
        return self.df_apply(inputs)

    def cleanup_pbar(self) -> None:
        try:
            self.pbar.close()
            self.pbar = None
        except AttributeError:
            pass

    def update_pbar(self):
        """Updates the progress bar, marking one more row as completed."""
        try:
            self.pbar.update(1)
        except AttributeError:
            pass

    # a more detailed type hint is possible, but it is messy,
    # the docstring explains the formats more clearly
    @staticmethod
    def _convert_results_from_old_format(result: Sequence) -> Sequence:
        """detects if results are in the old format, and re-writes them if needed.
        results in the new format are returned unchanged.
        A warning about using the old format will be emitted, if applicable.

        Old format: results have the type:
            Tuple[Tuple[values of objectives], Tuple[values of constraints]]
        New format: results have the type:
            Tuple[values of objectives, values of constraints]

        These formats are also both used for error_value

        :param result: the result in either old or new format
        :return: the result in the new format
        """
        if len(result) == 2:
            objectives, constraints = result
            if (
                isinstance(objectives, Sequence)
                and isinstance(constraints, Sequence)
                and all(
                    isinstance(x, (float, int))
                    for x in list(objectives) + list(constraints)
                )
            ):
                result = tuple(objectives) + tuple(constraints)
                warnings.warn(
                    "Evaluators have changed the format used for storing "
                    "outputs and constraints: "
                    "Old format: (output1, output2, ...), "
                    "(constraint1, constraint2, ...) "
                    "New format: (output1, output2, ..., "
                    "constraint1, constraint2, ...)",
                    FutureWarning,
                )
        return result

    # a type hint is possible, but it is messy,
    # the docstring explains the formats more clearly
    def _convert_error_mode_from_old_format(self, error_value):
        """detects if error_values are in the old format, and re-writes them if needed.
        error_values in the new format are returned unchanged.
        A warning about using the old format will be emitted, if applicable.

        :param error_value: the error value in either the old or new format
        :return: the error mode in the new format
        """
        warning = (
            "Evaluators have changed the format used for error values. "
            "The new format accepts either None or a tuple of values. "
            "See the evaluator docstring for details on the new format. "
            "\nYour error mode has automatically been converted to a new format equivalent."
        )
        if error_value is None:
            return error_value
        if tuple(error_value) == (None, None):
            if self.problem.num_outputs + self.problem.num_constraints == 2:
                return error_value
            else:
                warnings.warn(warning, FutureWarning)
                return [
                    None,
                ] * (self.problem.num_outputs + self.problem.num_constraints)
        if len(error_value) != 2:
            return error_value
        # the length 2 case is ambiguous, since we can have either 2 values
        # to use as the error mode values, or an objectives constraint pair
        # from the old style of input
        # if one of the values is a sequence we have an old style pair
        outputs, constraints = error_value
        if isinstance(outputs, Sequence) or isinstance(constraints, Sequence):
            warnings.warn(warning, FutureWarning)
            if outputs is None:
                outputs = [
                    None,
                ] * self.problem.num_outputs
            outputs = list(outputs)
            if constraints is None:
                constraints = [
                    None,
                ] * self.problem.num_constraints
            constraints = list(constraints)
            error_value = outputs + constraints
        return error_value


class EvaluatorGeneric(AbstractEvaluator):
    """Generic Evaluator

    This evaluator is a wrapper around a evaluation function. Can be useful for
    quick debugging.
    """

    eval_func_format = Callable[[Sequence], Tuple[float, ...]]

    def __init__(
        self,
        evaluation_func: eval_func_format,
        problem: Problem,
        error_mode: str = "Failfast",
        error_value: Sequence = None,
        progress_bar: bool = True,
    ):
        """
        :param evaluation_func: a function that takes as input an list of values,
                and gives as output a tuple of the objective values for that point in the solution space
        :param problem: description of the inputs and outputs the evaluator will use
        :param progress_bar: whether or not to display a progress bar
        """

        super().__init__(
            problem=problem,
            error_mode=error_mode,
            error_value=error_value,
            progress_bar=progress_bar,
        )
        self._evaluation_func = evaluation_func

    def eval_single(self, values: Sequence) -> Sequence:
        result = self._evaluation_func(values)
        result = self._convert_results_from_old_format(result)
        self.update_pbar()
        return result


@deprecated(
    version="1.6.0",
    reason="EvaluatorSR has been renamed as EvaluatorGeneric with same functionality.",
)
class EvaluatorSR(EvaluatorGeneric):
    """Surrogate Model Evaluator

    This evaluator has been replaced by EvaluatorGeneric, will be removed in a future release.
    """

    pass  # functionality is exactly the same as EvaluatorGeneric, no subclassing needed.


# TODO: Add an option/subclass that automatically bundles several single variable models into a multiobjective model.
class AdaptiveSR(AbstractEvaluator, ABC):
    """A Template for making adaptive sampling based models compatible with the
    evaluator interface.

    Wraps a user specified model training process
    Evaluates the current model
    Retrains the model on new data
    Records training data
    Clears cache when retrained
    Has a reference evaluator used as a ground-truth
    Optional:
    - Finds the best points to add to the model
    - Update the model without fully retraining
    TODO: make a version that can wrap a scikit-learn pipeline to reduce boilerplate
    TODO: make a version that can bundle multiple single objective models together
    """

    # helper functions provided by AdaptiveSR (Generally avoid editing these, but use them as needed)
    # append_data(X, y)
    # do_infill
    # get_from_reference

    # functions with defaults (These can be removed from this template if you like the defaults)
    # They may depend on some of the optional functions in order to work if using the defaults
    # __init__
    # infill -> get_infill, _update_model
    # update_model

    # optional functions (These will not work unless you implement them)
    # get_infill

    # required functions
    # train
    # eval_single

    tabular = Union[DF, np.array]

    def __init__(
        self,
        reference: AbstractEvaluator = None,
        error_mode: str = "Failfast",
        error_value: Sequence = None,
    ):
        self.reference: AbstractEvaluator = reference
        super().__init__(
            problem=reference.problem,
            error_mode=error_mode,
            error_value=error_value,
        )

        self.model = None
        self.data: DF = pd.DataFrame(
            columns=self.problem.names(parts=["inputs", "outputs", "constraints"])
        )

    @property
    def problem(self):
        return self.reference.problem

    @problem.setter
    def problem(self, value: Problem):
        self.reference.problem = value

    def append_data(self, data: tabular, deduplicate: bool = True) -> None:
        """Adds the X and y data to input_data and output_data respectively

        :param data: a table of training data to store
        :param deduplicate: whether to remove duplicates from the combined DataFrame
        :return:
        """
        self.cache_clear()  # TODO: decide on a consistent way of tracking this
        # can we assume users will only modify the data using this method or will call cache_clear themselves
        new_data = self.problem.to_df(data, ["inputs", "outputs", "constraints"])
        self.data = self.data.append(new_data, ignore_index=True)
        if deduplicate:
            self.data.drop_duplicates(inplace=True)

    def get_infill(self, num_datapoints: int) -> tabular:
        """Generates data that is most likely to improve the model, and can be used for retraining.

        :param num_datapoints: the number of datapoints to generate
        :return: the datapoints generated, in some tabular datastructure
        """
        raise NotImplementedError

    def do_infill(self, data: DF) -> None:
        """Updates the model using the inputs X and outputs y, and stores the added data

        :param data: a table of training data
        :return: None
        """
        old_df = self.data
        df, parts = self.problem.partial_df(
            data, parts=["inputs", "outputs", "constraints"]
        )
        if parts == ["inputs"]:
            outputs: DF = self.get_from_reference(df)
            df = pd.concat([df, outputs], axis=1)
        self.append_data(df)
        if self.model is None:
            self.train()
        else:
            self.update_model(df, old_df)

    def update_model(self, new_data: tabular, old_data: DF = None) -> None:
        """Modifies self.model to incorporate the new data.

        This function should not edit the existing data

        :param new_data: a table of inputs and outputs
        :param old_data: the table of inputs and outputs without the new data
        :return: None
        """
        self.train()

    def infill(self, num_datapoints: int) -> None:
        """Adds num_datapoints samples to the model and updates it.

        :param num_datapoints: number of datapoints to add to the model's training set
        :return: None
        """
        inputs: DF = self.problem.to_df(self.get_infill(num_datapoints), "inputs")
        outputs: DF = self.get_from_reference(inputs)
        self.do_infill(pd.concat([inputs, outputs], axis=1))

    @abstractmethod
    def train(self) -> None:
        """Generates a new model using the stored data, and stores it as self.model"""
        pass

    @abstractmethod
    def eval_single(self, values: Sequence, **kwargs) -> Tuple:
        """Evaluates a single input point

        :param values: The datapoint to evaluate
        :param kwargs: Arbitrary keyword arguments.
        :return: A tuple of the predicted outputs for this datapoint
        """
        pass

    def get_from_reference(self, X: tabular) -> DF:
        """Use the reference evaluator to get the real value of a dataframe of datapoints

        :param X: a table containing the datapoints to evaluate
        :return: a DataFrame containing the results of the datapoints
        """
        df = self.problem.to_df(X, "inputs")
        return self.reference.df_apply(df)


class EvaluatorEP(AbstractEvaluator, ABC):
    """This evaluator uses a Problem to modify a building, and then simulate it.
    It keeps track of the building and the weather file."""

    def __init__(
        self,
        problem: Problem,
        building,
        epw: PathLike = config.files["epw"],
        out_dir: PathLike = config.out_dir,
        err_dir: PathLike = config.err_dir,
        error_mode: str = "Failfast",
        error_value: Optional[Sequence] = None,
        version=None,
        progress_bar: bool = True,
        ep_path: PathLike = None,
        *,
        epw_file: PathLike = None,
    ):
        """

        :param problem: a parametrization of the building and the desired outputs
        :param building: the building that is being simulated.
        :param epw: path to the epw file representing the weather
        :param out_dir: the directory used for files created by the EnergyPlus simulation.
        :param err_dir: the directory where files from a failed run are stored.
        :param error_mode: One of {'Failfast', 'Silent', 'Print'}.
            Failfast: Any error aborts the evaluation.
            Silent: Evaluation will return the `error_value` for any lists of values that raise an error.
            Print: Same as silent, but warnings are printed to stderr for any errors.
        :param error_value: The value of the evaluation if an error occurs. Incompatible with error_mode='Failfast'.
        :param version: Deprecated
        :param progress_bar: whether or not to display a progress bar
        :param epw_file: Deprecated. Use epw instead. Path to the epw file representing the weather.
        """

        super().__init__(
            problem=problem,
            error_mode=error_mode,
            error_value=error_value,
            progress_bar=progress_bar,
        )
        if out_dir is None:
            self.out_dir = out_dir
        else:
            self.out_dir = Path(out_dir, config.out_wrapper)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        self.ep_path = ep_path
        self.err_dir = Path(err_dir, config.error_wrapper)

        self.building = building

        # backwards compatibility for renaming epw_file -> epw
        if epw_file and epw != config.files["epw"]:
            raise ValueError(
                "epw_file and epw cannot be used together. Just use epw, as epw file is deprecated."
            )
        elif epw_file:
            epw = epw_file
            warn(
                FutureWarning(
                    "epw_file has been deprecated and will be removed in the future. Use epw instead.",
                )
            )
        # backwards compatibility for removing version
        if version:
            warnings.warn(
                "the version argument is deprecated for run_building,"
                " and will be removed in the future",
                FutureWarning,
            )
            assert version == eplus.get_idf_version(building), "Incorrect version"

        if not (str(epw).endswith(".epw") and os.path.exists(epw)):
            raise ValueError(
                f"epw argument expects path to an epw file, file provided was: {epw}"
            )
        self.epw = epw

        # epw can be read when needed, instead of here
        self.epw_contents = None

    def df_apply(
        self,
        df: DF,
        keep_input=False,
        processes: int = 1,
        keep_dirs: bool = False,
        *,
        out_dir=config.out_dir,
    ) -> DF:
        """Applies this evaluator to an entire dataFrame, row by row.

        :param df: a DataFrame where each row represents valid input values for this Evaluator.
        :param keep_input: whether to include the input data in the returned DataFrame
        :param processes: amount of cores to use
        :param keep_dirs: whether or not keep output directory
        :return: Returns a DataFrame with one column containing the results for each objective.
        """
        temp_self = self.out_dir
        temp_out = out_dir
        if not keep_dirs:
            out_dir = self.out_dir = None

        result = super().df_apply(
            df,
            keep_input=keep_input,
            processes=processes,
            keep_dirs=keep_dirs,
            out_dir=out_dir,
        )
        # TODO: Rework automatic column naming
        if keep_dirs:
            result.rename(
                columns={result.columns[-1]: "output_dir"}, inplace=True, errors="raise"
            )
        if not keep_dirs:
            self.out_dir = temp_self
            out_dir = temp_out

        self.cleanup_pbar()
        return result

    def eval_single(self, values: Sequence, out_dir=None, keep_dirs=False):
        out_dir = out_dir or self.out_dir
        err_dir = self.err_dir
        if keep_dirs:
            err_dir = out_dir = ef.generate_dir(out_dir)
        current_building = self._generate_building_from_row(values)
        results = eplus.run_building(
            current_building,
            out_dir=out_dir,
            err_dir=err_dir,
            epw=self.epw,
            error_mode=self.error_mode,
            ep_path=self.ep_path,
        )
        outputs = tuple((objective(results) for objective in self.problem.outputs))
        constraints = tuple(
            (constraint(results) for constraint in self.problem.constraints)
        )
        extracted_results = outputs + constraints
        if self.problem.add_outputs_list is not None:
            self.problem.record_results(values, results)
        self.update_pbar()
        if keep_dirs:
            return extracted_results + (out_dir,)
        else:
            return extracted_results

    @property
    def building(self):
        return self._building

    @building.setter
    def building(self, new_building) -> None:
        """Changes the building simulated.
        Changing this resets the cache.

        :param new_building: the building to use
        :return: None
        """
        for io in self.problem:
            io.setup(new_building)
        self.cache_clear()
        self._building = new_building

    @property
    def epw(self) -> PathLike:
        return self._epw

    @epw.setter
    def epw(self, value: PathLike) -> None:
        """Changes the epw file used with this building.
        Changing this resets the cache.

        :param value: path to the new epw file to use.
        :return: None
        """
        self.cache_clear()
        self._epw = value

    # REMOVE?
    def generate_building(self, df: DF, index: int, file_name: str) -> None:
        """generate idf file

        :param df: dataFrame of the select row.
        :param index: start point.
        :param file_name: file name used to save as.
        :return: None
        """
        l = df.values.tolist()[index][: len(self.problem.value_descriptors)]
        current_building = self._generate_building_from_row(l)
        if isinstance(current_building, dict):
            with open(f"{file_name}.epJSON", "w") as fp:
                json.dump(current_building, fp)
        else:
            current_building.saveas(f"{file_name}.idf")

    # This "apply all of the parameters" behaviour is duplicated in at least 2 other places.
    # EvaluatorEH.eval_single and pyehub_funcs.pyehub_parameter_editor
    def _generate_building_from_row(self, row):
        """Generates a copy of this evaluator's building from a row of data"""
        self.validate(row)
        current_building = copy.deepcopy(self.building)
        values = {
            descriptor: value
            for descriptor, value in zip(self.problem.value_descriptors, row)
        }
        # We could verify if row has column names,
        # and check that those match, warning if there is a mismatch.
        # This could also be done in validate()
        for parameter in self.problem.parameters:
            parameter.transformation_function(current_building, values)
        return current_building


class EvaluatorEH(AbstractEvaluator, ABC):
    """This evaluator uses a Problem to modify an energy hub, and then solve it."""

    def __init__(
        self,
        problem: Problem,
        hub,
        out_dir: PathLike = config.out_dir,
        err_dir: PathLike = config.err_dir,
        error_mode: str = "Failfast",
        error_value: Optional[Sequence] = None,
        progress_bar: bool = True,
    ):
        """

        :param problem: a parametrization of the hub and the desired outputs
        :param hub: the energy hub that is being simulated.
        :param out_dir: the directory used for files created by the PyEHub simulation.
        :param err_dir: the directory where files from a failed run are stored.
        :param error_mode: One of {'Failfast', 'Silent', 'Print'}.
            Failfast: Any error aborts the evaluation.
            Silent: Evaluation will return the `error_value` for any lists of values that raise an error.
            Print: Same as silent, but warnings are printed to stderr for any errors.
        :param error_value: The value of the evaluation if an error occurs. Incompatible with error_mode='Failfast'.
        :param progress_bar: whether or not to display a progress bar
        """

        super().__init__(
            problem=problem,
            error_mode=error_mode,
            error_value=error_value,
            progress_bar=progress_bar,
        )
        self.out_dir = out_dir
        self.err_dir = err_dir
        self.hub = hub
        self.problem = problem
        self.config_settings = config.solver_settings

    # override of validate due to pyehub model inputs being lists
    def validate(self, values: Sequence) -> None:
        if len(values) != self.problem.num_inputs:
            raise ValueError(
                f"Wrong number of input values."
                f"{len(values)} provided, {self.problem.num_inputs} expected"
            )

    def eval_single(self, values: Sequence) -> tuple:
        current_hub = copy.deepcopy(self.hub)

        # this is duplicated in EvaluatorEP._generate_building_from_row
        self.validate(values)
        value_pairs = {
            descriptor: value
            for descriptor, value in zip(self.problem.value_descriptors, values)
        }
        for parameter in self.problem.parameters:
            parameter.transformation_function(current_hub, value_pairs)

        objectives = self.problem.names("outputs")
        current_hub.compile()
        primary_objective = objectives[0]
        current_hub.objective = primary_objective

        results = current_hub.solve()

        if self.out_dir is not None:
            output_file = self.config_settings["output_file"]
            self.out_dir = self.out_dir / output_file
            output_excel(
                results["solution"],
                self.out_dir,
                time_steps=len(current_hub.time),
                sheets=["other", "capacity_storage", "capacity_tech"],
            )

        outputs = tuple(
            (
                pf.get_by_path(results["solution"], [objective])
                for objective in objectives
            )
        )
        if self.problem.add_outputs_list is not None:
            self.problem.record_results(values, results)
        self.update_pbar()

        return outputs
