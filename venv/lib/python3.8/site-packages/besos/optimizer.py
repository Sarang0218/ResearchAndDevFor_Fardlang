"""
The optimizer module provides wrappers which connect optimization packages to Evaluators.
Currently we support `platypus <https://github.com/Project-Platypus/Platypus>`_
and `rbf_opt <https://github.com/coin-or/rbfopt>`_

The supported Platypus algorithms are:

- GeneticAlgorithm
- EvolutionaryStrategy
- NSGAII
- EpsMOEA
- GDE3
- SPEA2
- MOEAD
- NSGAIII
- ParticleSwarm
- OMOPSO
- SMPSO
- CMAES
- IBEA
- PAES
- PESA2
- EpsNSGAII

These algorithms have the same configuration options as their counterparts in platypus,
with the parts provided by besos evaluators filled automatically.
"""

# Python Core Libraries
import os
import sys
from collections import OrderedDict
from functools import partial
from inspect import signature, Parameter, Signature
from typing import Type, List

# External Libraries
import dask_utils
import numpy as np
import pandas as pd
import platypus
import rbfopt

from besos.config import optimizer as conf
from besos.evaluator import AbstractEvaluator
from besos.parameters import CategoryParameter, RangeParameter
from besos.problem import Problem


# TODO: evaluations or end criterion?
def platypus_alg(
    evaluator: AbstractEvaluator,
    algorithm: Type[platypus.Algorithm],
    evaluations: int = conf["evaluations"],
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Uses a platypus algorithm to optimise over an evaluator.

    :param evaluator: An evaluation function to optimise over.
    :param algorithm: The platypus algorithm to use.
    :param evaluations: The algorithm will be stopped once it uses more than this many evaluations.
    :param args: arguments to pass to `algorithm`
    :param kwargs: keyword arguments to pass to `algorithm`.
    :return: the non-dominated solutions found by the algorithm.
    """

    problem = evaluator.to_platypus()
    evaluator.problem.pre_optimisation()
    alg: platypus.AbstractGeneticAlgorithm = algorithm(problem, *args, **kwargs)
    alg.run(evaluations)

    # TODO: Currently we use the default value of parts when calling solutions_to_df
    # can we find a better default for this function?
    return solutions_to_df(platypus.unique(alg.result), evaluator.problem)


_ordered_argtypes = [
    (Parameter.POSITIONAL_OR_KEYWORD, False),
    (Parameter.POSITIONAL_OR_KEYWORD, True),
    (Parameter.VAR_POSITIONAL, False),
    (Parameter.KEYWORD_ONLY, False),
    (Parameter.KEYWORD_ONLY, True),
    (Parameter.VAR_KEYWORD, False),
]


def get_operator(problem: platypus.Problem, mutation=False):
    """Creates a valid operator for the given platypus Problem.
    This is provided to help when optimizing over problems with mixed types.

    :param problem: the platypus problem this operator should apply to
    :param mutation: if True create a mutation operator, if False create a variation operator
    :return:
    """
    operators = []
    if mutation:
        defaults = platypus.config.PlatypusConfig.default_mutator
        class_ = platypus.CompoundMutation
    else:
        defaults = platypus.config.PlatypusConfig.default_variator
        class_ = platypus.CompoundOperator
    for t in problem.types:
        if t in defaults:
            operators.append(defaults[t])
        else:
            for super_t, operator in defaults.items():
                if isinstance(t, super_t):
                    operators.append(operator)
    if mutation:
        return class_(*operators)
    return class_(*operators)


def _alg_t(algorithm):
    """A function that wraps a platypus algorithm in a common workflow.
    This function propagates the algorithm's function signature out to produce a more descriptive
    function signature."""

    # This is perhaps an overly fancy way of doing it, but it avoids having to
    # do the same process manually for all platypus algorithms, and will
    # update the signatures automatically if they change in platypus.

    defaults = {
        "variator": get_operator,
        "mutator": partial(get_operator, mutation=True),
    }

    def template(evaluator, evaluations=conf["evaluations"], *args, **kwargs):
        for kwarg, default in defaults.items():
            if kwarg not in kwargs or kwargs[kwarg] == "automatic":
                kwargs[kwarg] = default(evaluator.to_platypus())
        return platypus_alg(evaluator, algorithm, evaluations, *args, **kwargs)

    temp_name = (
        # (function, arguments_to_omit)
        (template, ["args", "kwargs"]),
        (algorithm.__init__, ["self", "problem"]),
    )

    argdict = {kind: [] for kind in _ordered_argtypes}
    for i, (func, omit) in enumerate(temp_name):
        sig = signature(func)
        p: Parameter
        for name, p in sig.parameters.items():
            if name in omit:
                continue
            if i > 0:
                # convert non-default keyword arguments to keyword only arguments
                if (
                    p.kind is Parameter.POSITIONAL_OR_KEYWORD
                    and p.default is Parameter.empty
                ):
                    p = p.replace(kind=Parameter.KEYWORD_ONLY)
                # use the defaults specified above
                if name in defaults:
                    p = p.replace(default="automatic")

            argdict[(p.kind, p.default is not Parameter.empty)].append(p)
    arglist = []
    for kind, default in _ordered_argtypes:
        arglist.extend(argdict[kind, default])

    template.__signature__ = Signature(
        parameters=arglist, return_annotation=signature(template).return_annotation
    )
    template.__name__ = algorithm.__name__
    return template


# The following functions are convenience wrappers for platypus' implementations
# of the algorithms with the same name
# they return the non-dominated solutions found by the algorithm

GeneticAlgorithm = _alg_t(platypus.GeneticAlgorithm)
EvolutionaryStrategy = _alg_t(platypus.EvolutionaryStrategy)
NSGAII = _alg_t(platypus.NSGAII)
EpsMOEA = _alg_t(platypus.EpsMOEA)
GDE3 = _alg_t(platypus.GDE3)
SPEA2 = _alg_t(platypus.SPEA2)
MOEAD = _alg_t(platypus.MOEAD)
NSGAIII = _alg_t(platypus.NSGAIII)
ParticleSwarm = _alg_t(platypus.ParticleSwarm)
OMOPSO = _alg_t(platypus.OMOPSO)
SMPSO = _alg_t(platypus.SMPSO)
CMAES = _alg_t(platypus.CMAES)
IBEA = _alg_t(platypus.IBEA)
PAES = _alg_t(platypus.PAES)
PESA2 = _alg_t(platypus.PESA2)
EpsNSGAII = _alg_t(platypus.EpsNSGAII)


def values_to_solution(
    values: list, problem: platypus.Problem, evaluate=True
) -> platypus.Solution:
    """Produces an unevaluated platypus solution with the given inputs

    :param values: a list of input values to convert to a platypus solution
    :param problem: the platypus problem this solution applies to
    :param evaluate: whether to evaluate the solution before returning (solution will be unevaluated otherwise)
    :return: the corresponding platypus.Solution object
    """
    solution = platypus.Solution(problem)
    if len(values) != problem.nvars:
        raise ValueError(
            f"Length of values is {len(values)} but expected length of {problem.nvars}"
        )
    solution.variables = list(values)
    if evaluate:
        solution.evaluate()
    return solution


# TODO: Unify `parts` handling between this and Problem to remove duplicate code if possible


def solution_to_values(solution: platypus.Solution, parts="all") -> list:
    """Converts a platypus solution to a list containing the same values

    :param solution: a platypus solution to convert
    :param parts: which parts of the solution should be kept
    :return: a list of the requested values from the solution
    """
    part_options = OrderedDict(
        inputs="variables",
        outputs="objectives",
        constraints="constraints",
        violation="violation",
    )
    if parts == "all":
        parts = part_options.keys()
    assert set(parts) <= set(
        part_options
    ), f'parts must be a subset of {set(part_options.keys())} or "all"'
    result = []
    for part in parts:
        if part == "violation":
            result.append(solution.constraint_violation)
        elif part == "inputs":
            encoded = getattr(solution, part_options[part])
            decoded = [
                plat_type.decode(val)
                for val, plat_type in zip(encoded, solution.problem.types)
            ]
            result.extend(decoded)
        else:
            result.extend(getattr(solution, part_options[part]))
    return result


def solutions_to_df(
    solutions: List[platypus.Solution], problem, parts="all", flag_optimal=True
) -> pd.DataFrame:
    """Converts a list of platypus solutions to a DataFrame, with one row corresponding to each solution

    :param solutions: list of solutions to convert
    :param problem: the column names for DataFrame
    :param parts: which parts of the solutions should be kept
    :param flag_optimal: whether to include a boolean column denoting whether each solution is pareto-optimal
    :return: a DataFrame
    """

    def to_col_vals(solution_list):
        return list(
            zip(*(solution_to_values(solution, parts) for solution in solution_list))
        )

    solutions = platypus.unique(solutions)
    non_dominated = platypus.nondominated(solutions)
    columns = problem.names(parts)
    values, non_dom_vals = to_col_vals(solutions), to_col_vals(non_dominated)
    assert len(columns) == len(
        values
    ), f"{len(values)} values does not match {len(columns)} columns"
    # TODO: Intuit the dataframe column types based on the types of the parameters of the problem
    # or use the to_df method of the problem object
    solution_df = pd.DataFrame(
        {column: data for column, data in zip(columns, values)}
    )  # , dtype=float
    if flag_optimal:
        non_dom_df = pd.DataFrame(
            {column: data for column, data in zip(columns, non_dom_vals)}
        )  # , dtype=float
        df = pd.merge(solution_df, non_dom_df, how="outer", indicator="pareto-optimal")
        df["pareto-optimal"] = df["pareto-optimal"] == "both"
    else:
        df = solution_df
    df = problem.overwrite_df(df)
    problem.post_optimazation()
    return df


def df_solution_to_solutions(
    df: pd.DataFrame,
    platypus_problem: platypus.Problem,
    besos_problem: Problem,
) -> List[platypus.Solution]:
    """Converts a solution DataFrame to a list of platypus solutions, with each row converted to one solution

    :param df: DataFrame to convert
    :param platypus_problem: platypus problem that the solutions apply to
    :param besos_problem: besos problem that the solutions apply to
    :return: a list of platypus solutions
    """
    input_df = df.copy(deep=True)
    labels = df.columns.values.tolist()
    for input in besos_problem.inputs:
        labels.remove(input.name)
    for label in labels:
        input_df.drop(columns=label, inplace=True)
    return list(input_df.apply(values_to_solution, problem=platypus_problem, axis=1))


def df_to_solutions(
    df: pd.DataFrame, problem: platypus.Problem
) -> List[platypus.Solution]:
    """Converts a DataFrame to a list of platypus solutions, with each row converted to one solution

    :param df: DataFrame to convert
    :param problem: platypus problem that the solutions apply to
    :return: a list of platypus solutions
    """
    return list(df.apply(values_to_solution, problem=problem, axis=1))


# rbf opt

# TODO: Add support for the other termination criterion
def rbf_opt(
    evaluator, evaluations, hide_output: bool = True, bonmin_path=None, rand_seed=None
):
    """
    This is wrapper to the rbfopt function.

    :param evaluator: An evaluation function to optimise over.
    :param evaluations: The algorithm will be stopped once it uses more than this many evaluations.
    :param hide_output: Whether to suppress output from the rbf_opt algorithm
    :param bonmin_path: The path to Bonmin, a specific feature useful for the cluster.
    :param rand_seed: Give a seed number to make sure the random has numbers.
    :return: a list of solutions

    """

    # default setting from rbfopt or do something custom (specifically for the cluster)
    # This parameter points rbfopt to the solver it was written to use.
    if bonmin_path == None:
        bonmin_path_set = "bonmin"
    else:
        bonmin_path_set = bonmin_path
    # This parameter seeds the random number generator of numpy.
    # To do a proper assessment this needs to be randomly set by the user.
    # else every run will just do the same search.
    if rand_seed == None:
        rand_seed_set = 937627691
    else:
        rand_seed_set = rand_seed

    problem = evaluator.problem
    inputs = problem.value_descriptors
    dimension = problem.num_inputs
    num_outputs = problem.num_outputs

    assert (
        num_outputs == 1
    ), f"This is a single objective optimizer, but you have {num_outputs} objectives"
    assert problem.num_constraints == 0, "This optimizer cannot handle constraints"
    # may throw an attribute error if invalid parameters are used (i.e categorical)
    param_attrs = []
    categoryparams = {}

    for value_descriptor in inputs:
        if isinstance(
            value_descriptor, CategoryParameter
        ):  # editing the CategoryParameter to work for rbfopt
            categoryparams[
                inputs.index(value_descriptor)
            ] = value_descriptor  # saving parameter with index for later
            param_attrs.append((0, len(value_descriptor.options) - 1, "I"))
        else:
            param_attrs.append(
                (value_descriptor.min, value_descriptor.max, value_descriptor.rbf_type)
            )

    var_lower, var_upper, var_type = (np.array(x) for x in zip(*param_attrs))
    # with the current classes var_type is just an array of 'R's
    # this would change if we add a descriptor for integers
    print(var_lower, var_upper, var_type)

    def wrapper(values):
        values_temp = values.copy().tolist()
        for index, item in categoryparams.items():
            values_temp[index] = item.options[int(values_temp[index])]
        return evaluator(values_temp)[0]

    def run_alg():
        bb = rbfopt.RbfoptUserBlackBox(
            dimension=dimension,
            var_lower=var_lower,
            var_upper=var_upper,
            var_type=var_type,
            obj_funct=wrapper,
        )

        if evaluations * 10 < 1000:  # 1000 is the default of max_iterations,
            iterations = (
                evaluations * 10
            )  # 10 is randomly chosen to give plenty of space for proper optimization
        else:
            iterations = 1000

        settings = rbfopt.RbfoptSettings(
            max_iterations=iterations,  # NOTE: Placed for CategoryParameter infinite loop problem with floats
            max_evaluations=evaluations,
            minlp_solver_path=bonmin_path_set,
            rand_seed=rand_seed_set,
        )
        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        return alg.optimize()

    if hide_output:
        with open(os.devnull, "w") as null:
            old, sys.stdout = sys.stdout, null
            try:
                val, x, itercount, evalcount, fast_evalcount = run_alg()
            finally:
                sys.stdout = old
    else:
        val, x, itercount, evalcount, fast_evalcount = run_alg()

    x = list(x)
    for index, item in categoryparams.items():
        x[index] = item.options[int(x[index])]

    return problem.to_df(np.array([np.concatenate([x, [val]])]))
