"""
Classes used to bundle the parameters, objectives and constraints,
and to manage operations that involve all of them at once,
such as converting data related to the problem to a DataFrame.
"""

# Python Core Libraries
import warnings
from typing import Union, List, Dict, Callable, Iterable

# External Libraries
import numpy as np
import pandas as pd
import platypus

# BESOS Imports
from besos import IO_Objects
from besos import config
from besos import objectives
from besos import parameters
from deprecated.sphinx import deprecated


# TODO: Consider storing the constraint bounds with the constraints themselves, not in the problem
# also consider storing the direction of optimisation inside the objectives
# might be able to inherit some of the constraint parsing from platypus, not sure if that is worth the hassle
class Problem(IO_Objects.ReprMixin):
    """
    A class that collects all of the inputs, outputs and constraints related to a model.
    Problems track what inputs are valid, and how to apply those inputs to a model.

    It tracks constraint bounds.
    Automatically converts certain shortcut notation:
    - Strings become name only Parameters
    - Integers become that many numbered Parameters
    Gives access to names of all parts of the problem
    Resolves duplicate names
    Converts numpy arrays to a DataFrame matching the format of the problem or some
    combination pieces of the format (ex: only inputs and constraints)
    Can convert to a platypus Problem (which lacks an evaluation function).
    """

    valid_parts: List[str] = ["inputs", "outputs", "constraints", "violation"]
    default_converters = {
        "outputs": IO_Objects.Objective,
        "constraints": IO_Objects.Objective,
    }

    def __init__(
        self,
        inputs: Union[int, List[Union[str, parameters.Parameter]]] = None,
        outputs: Union[int, List[Union[str, IO_Objects.Objective]]] = None,
        constraints: Union[int, List[Union[str, IO_Objects.Objective]]] = None,
        add_outputs: Union[int, List[Union[str, IO_Objects.Objective]]] = None,
        *,
        constraint_bounds: List[str] = None,
        minimize_outputs: List[bool] = None,
        converters: Dict[str, Callable[[str], IO_Objects.IOBase]] = None,
    ):
        """
        :param inputs: A list of Parameters, or an integer.
            If a list is used, strings are converted to Parameters and this list determines the valid inputs.
            If an integer, this problem accepts that many inputs.
        :param outputs: A list of Objectives, or an integer.
            If a list is used, strings are converted to Objectives and this list determines the valid inputs.
            If an integer, this problem requires that many outputs
        :param constraints: A list of Objectives to be used as constraints, or an integer.
            If a list is used, strings are converted to Objectives and this list determines the valid inputs.
            If an integer, this problem requires that many constraints.
        :param add_outputs: Outputs that don't need to be optimized in optimization algorithm
        :param constraint_bounds: a list of platypus-style constraint bounds, such as "<=750".
            These are used when converting constraints for use with platypus.
            Check the platypus documentation for more details.
        :param minimize_outputs: A list with true/false values corresponding to each output.
            Outputs having a corresponding value of True will be minimized, while outputs
            having a corresponding value of False will be maximized instead.
        :param converters: A dictionary with keys from {"outputs", "constraints"} where values indicate how to
            convert those kinds of values to appropriate objectives/constraints for this problem.
        """
        super().__init__()
        # TODO remove self.converters, use as an argument instead
        self.converters = converters or self.default_converters
        extra_keys = set(self.converters.keys()) - set(self.valid_parts)
        if extra_keys:
            raise ValueError(
                f"The keys {extra_keys} are not valid for this Problem. Only {self.valid_parts} are valid"
            )

        self.parameters = self._io_to_list(inputs, "inputs")
        self.value_descriptors = self._get_descriptors(self.parameters)

        self.outputs = self._io_to_list(outputs, "outputs")
        self.add_outputs = self._io_to_list(add_outputs, "outputs")
        self.add_outputs_list = None

        # TODO: Move this information to the output objects
        self.minimize_outputs = minimize_outputs or [True] * self.num_outputs
        msg = "outputs and minimize_outputs must have the same length"
        assert len(self.minimize_outputs) == self.num_outputs, msg

        self.constraints = self._io_to_list(constraints, "constraints")
        # TODO: consider using platypus's constraints here
        self.constraint_bounds = constraint_bounds or []
        msg = "constraints and constraint_bounds must have the same length"
        assert len(self.constraint_bounds) == self.num_constraints, msg
        self.fix_names()
        self._add_repr("inputs", "parameters", check=True)
        self._add_reprs(
            [
                "outputs",
                "minimize_outputs",
                "constraints",
                "constraint_bounds",
                "converters",
            ],
            check=True,
        )

    @staticmethod
    def _get_descriptors(param_list: List[parameters.Parameter]):
        found = set()
        ordered_descriptors = []
        for parameter in param_list:
            for descriptor in parameter.value_descriptors:
                if descriptor not in found:
                    ordered_descriptors.append(descriptor)
                    found.add(descriptor)
        return ordered_descriptors

    def fix_names(self):
        mapping = {}
        duplicates = []
        for obj in self.value_descriptors + self.outputs + self.constraints:
            mapping[obj.name] = mapping.get(obj.name, []) + [obj]
        for name, objects in mapping.items():
            if len(objects) != 1:
                duplicates.append((name, objects))
        duplicates_2 = []
        for name, objects in duplicates:
            edited_names = []
            for i, obj in enumerate(objects):
                try:  # try block in case of no selector initialized
                    obj.name = f"{obj.name}_{obj.selector.class_name}"
                except:
                    pass
                edited_names.append(obj.name)
                if edited_names.count(obj.name) > 1:
                    duplicates_2.append(obj)
        if duplicates_2:
            warnings.warn(
                RuntimeWarning(
                    f"Duplicate names found. (duplicate, repetitions): "
                    f"{[(name, len(objects)) for name, objects in duplicates]}"
                    f"\nAttempting to fix automatically"
                )
            )
        for i, obj in enumerate(duplicates_2):
            obj.name = f"{obj.name}_{(i+1)}"

    def _io_to_list(self, io_objects: Union[int, List[IO_Objects.IOBase], None], part):
        """Converts a list of objects to a standard form:
        numbered placeholders, original datatype or io_object that match the part provided.
        """
        if io_objects is None:
            return []
        if isinstance(io_objects, int):
            if part == "inputs":
                # this repeats the default value for a parameter with no input specified
                # and should probably be refactored.
                def constructor(name):
                    return parameters.Parameter(
                        value_descriptors=IO_Objects.AnyValue(name=name)
                    )

            elif part in ["outputs", "constraints"]:
                constructor = IO_Objects.Objective
            else:
                raise ValueError(f"Cannot produce dummy values for part {part}")
            return [constructor(name=f"{part}_{i}") for i in range(io_objects)]
        if isinstance(io_objects, (str, IO_Objects.IOBase, parameters.Parameter)):
            io_objects = [io_objects]
        return [self.convert(o, part) for o in io_objects]

    def convert(self, io_object, part) -> IO_Objects.IOBase:
        """

        :param io_object: An object that should be converted to a parameter, objective or constraint
        :param part: one of 'inputs', 'outputs' or 'constraints' describing what to convert `io_object` to
        :return: the converted object
        """
        if isinstance(io_object, IO_Objects.IOBase):
            return io_object
        if part in self.converters:
            f = self.converters[part]
            try:
                return f(io_object)
            except TypeError as e:
                try:
                    if isinstance(io_object, dict):
                        return f(**io_object)
                    if isinstance(io_object, Iterable):
                        return f(*io_object)
                except:
                    pass
                raise TypeError(f"Cannot convert {io_object} to {part}") from e
        return io_object

    def expand_parts(self, parts: Union[str, List[str]]) -> List[str]:
        """Expands 'auto' and 'all' to the correct lists of parts, and wraps single parts in a list"""
        if parts == "auto":
            if self.num_constraints == 0:
                parts = ["inputs", "outputs"]
            else:
                parts = "all"
        if parts == "all":
            parts = self.valid_parts
        elif isinstance(parts, str):
            parts = [parts]

        if not set(parts) <= set(self.valid_parts):
            raise ValueError(
                f"parts must be a subset of {self.valid_parts + ['all']}, not {parts}"
            )
        return parts

    def names(self, parts: Union[str, List[str]] = "auto") -> List[str]:
        """

        :param parts: one of {'inputs', 'outputs', 'constraints', 'violation', 'all', 'auto'}
        :return: the names requested
        """
        parts = self.expand_parts(parts)
        names = []
        for attr in parts:
            if attr == "inputs":
                attr = "value_descriptors"
            if attr == "violation":
                names.append("violation")
            else:
                part = getattr(self, attr)
                if part is None:
                    raise ValueError(f"{attr} names not available")
                names.extend(IO_Objects.get_name(i) for i in part)
        return names

    # TODO: Add support for pareto-optimal column
    # TODO: Consolidate the different to_df code (ie from optimizer.py)
    def to_df(
        self,
        table: Union[np.array, pd.DataFrame],
        parts: Union[str, List[str]] = "auto",
    ) -> pd.DataFrame:
        """Converts the given table to a DataFrame that matches this problem's input/output format

        :param table: a table to be converted to a DataFrame. Must have the right number of columns.
        :param parts: inputs, outputs, constraints or all, depending on which data the DataFrame contains
        :return: A DataFrame containing the same data as the original table.
        """
        columns = self.names(parts)
        types = [getattr(p, "pd_type", None) for p in self.expand_parts(parts)]
        if isinstance(table, pd.DataFrame):
            if len(table.columns) != len(columns):
                raise ValueError(
                    f"columns: {columns} requested but {list(table.columns)} found"
                )
            return table[columns]

        df = pd.DataFrame(table, columns=columns)
        # TODO: Make the categorical columns have the type category instead of object (attempt commented out below)
        # for col, type_ in zip(df, types):
        #     if type_:
        #         df[col] = df[col].astype(type_)
        return df

    def partial_df(self, table: Union[np.array, pd.DataFrame], parts="all"):
        parts = self.expand_parts(parts)
        for i in range(1, len(parts) + 1):
            partial_parts = parts[:i]
            try:
                return self.to_df(table, partial_parts), partial_parts
            except ValueError:
                continue
        raise ValueError("Could not find a matching DataFrame")

    def to_platypus(self) -> platypus.Problem:
        """Converts this problem to a platypus problem.
        No evaluator will be included.

        :return: A corresponding platypus problem
        """
        problem = platypus.Problem(
            self.num_inputs, self.num_outputs, self.num_constraints
        )
        for i, descriptor in enumerate(self.value_descriptors):
            problem.types[i] = descriptor.platypus_type
        for i, direction in enumerate(self.minimize_outputs):
            problem.directions[i] = (
                platypus.Problem.MINIMIZE if direction else platypus.Problem.MAXIMIZE
            )
        for i, bound in enumerate(self.constraint_bounds):
            problem.constraints[i] = bound
        return problem

    def pre_optimisation(self):
        """Prepare optimisation for non objectives."""
        if self.add_outputs:
            self.add_outputs_list = []

    def record_results(self, inputs, results):
        """Record add_outputs results"""
        inputs = list(inputs)
        for obj in self.add_outputs:
            inputs.append(obj(results))
        self.add_outputs_list.append(inputs)

    def get_non_objective(self, df):
        """Create a list for dataframe with add_outputs value"""
        l = []
        for row in df.values.tolist():
            for obj in self.add_outputs_list:
                for i in range(self.num_inputs):
                    if row[i] != obj[i]:
                        break
                    if i == self.num_inputs - 1:
                        for j in range(len(self.add_outputs)):
                            if len(l) < j + 1:
                                l.append([])
                            l[j].append(obj[i + j + 1])
        return l

    def overwrite_df(self, df):
        """Insert add_outputs' values to dataframe"""
        if self.add_outputs:
            l = self.get_non_objective(df)
            for i, data in enumerate(l):
                df.insert(
                    len(df.columns),
                    self.add_outputs[i].name,
                    data,
                )
        return df

    def post_optimazation(self):
        self.add_outputs_list = None

    @property
    def num_inputs(self):
        return len(self.value_descriptors)

    @property
    def num_outputs(self):
        return len(self.outputs)

    @property
    def num_constraints(self):
        return len(self.constraints)

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self.inputs == other.inputs
            and self.outputs == other.outputs
            and self.constraints == other.constraints
        )

    def __iter__(self):
        return iter(self.parameters + self.outputs + self.constraints)

    @property
    @deprecated(
        version="2.0",
        reason="Problem.inputs is ambiguous, use .value_descriptors or .parameters instead.",
    )
    def inputs(self):
        return self.parameters


# TODO: consider having shortcuts for the converters instead of making this a whole different class
class EPProblem(Problem):
    """
    A problem with defaults that are appropriate for EnergyPlus simulations

    Strings for objectives/constraints become a MeterReader for the meter with
    that name.

    Integers still become numbered Parameters.
    """

    default_converters = {
        "outputs": objectives.MeterReader,
        "constraints": objectives.MeterReader,
    }

    def __init__(
        self,
        inputs=None,
        outputs=config.objectives,
        constraints=None,
        converters=None,
        **kwargs,
    ):
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
            converters=converters,
            **kwargs,
        )


class EHProblem(Problem):
    """A problem that works with PyEHub models"""

    # TODO: Restructure if possible to be more like other problems.
    def __init__(
        self,
        inputs=None,
        outputs=["total_cost"],
        constraints=None,
        converters=None,
        **kwargs,
    ):
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
            converters=converters,
            **kwargs,
        )

    # Overwritten functions to work with EvaluatorEH:
    def fix_names(self):
        pass
