"""
Helper functions for interacting with PyEHub.
"""

# Python Core Libraries
import operator
from functools import reduce
from typing import List

# External Libraries
import numpy as np
import pandas as pd
import pyehub
from deprecated import deprecated
from pyehub.energy_hub.ehub_model import EHubModel

# BESOS Imports
from besos import config


"""This file is a collection of functions to interact with PyEHub."""

# TODO: replace back with yaml loader?
def get_xlsx(excel_file: str = config.files.get("xlsx")):
    return excel_file


def get_hub():
    """Generates the base PyEHub model from the excel file."""
    # could check the settings to see what type of ehub model to create
    data_file = get_xlsx()
    model = pyehub.energy_hub.EHubModel(excel=data_file)
    return model


def get_by_path(root, items):
    """Access a nested object in root by item sequence. Used to navigate PyEHub's not flat dict."""
    # TODO: inset try statement here iscase the parameter doesn't exist to fail gracefully
    return reduce(operator.getitem, items, root)


def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence. Used to set values in PyEHub's not flat dict."""
    get_by_path(root, items[:-1])[items[-1]] = value


# TODO: Is this really needed? Can we just use the transformation function?
@deprecated(
    version="2.0.0",
    reason="Function re-implements the evaluator's set phase, "
    "and cannot handle parameters with multiple inputs.",
)
def pyehub_parameter_editor(
    hub, parameters: List["parameters.Parameter"], values: list
):
    """Changes the __dict__ of the energy hub for the parameters specified
    on initialization with the values given to evaluator."""
    for (parameter, value) in zip(parameters, values):
        parameter.selector.set(hub, value)


def _split_EP_days(hourly_series):
    """
    Splits the hourly series of EP simulations that simulate multiple days in different seasons into specific energyhub time series.
    """
    # TODO: replace with a better loop of checking for the number of hours in the series and splitting better
    result1 = hourly_series.head(24)
    result1 = result1.reset_index()
    result2 = hourly_series.tail(24)
    result2 = result2.reset_index()
    return result1, result2

    # TODO:


def ep_to_eh(input, index_size):
    """
    Converts EnergyPlus evaluator outputs to PyEHub compatible inputs.
    """
    # TODO: Needs to be finished

    input_dict = input.to_dict()

    # Need to have a better joining on index for the pandas dataframe instead of appending
    # (difficulties to be noted with inserting dictionaries into cells)
    columnnames = []
    for j in input_dict:
        columnnames.append(j)

    # 2x for the number of seasons in the splitting will need to have some value stored from
    # the smarter splitting or make an easy spot to modify a value that propigates through the whole system.
    # if using the input for the function
    # index_size *=2
    # which should be something that does this
    # index_size = len(EPdf)*2
    # df_input = pd.DataFrame(np.nan, index=range(0,index_size), columns=columnnames)
    df_input = pd.DataFrame(columns=columnnames)

    for objective in input_dict:
        for series in input_dict[objective]:
            result = input_dict[objective][series].to_frame()
            # TODO: make a more genaric splitting process
            cold_result, warm_result = _split_EP_days(result)

            # TODO: make a more generic unit conversion function
            # 3.6 Megajoules per kWh
            cold_result = cold_result / 3600000
            warm_result = warm_result / 3600000
            cold_dict = cold_result.to_dict()
            cold_dict = cold_dict["Value"]
            warm_dict = warm_result.to_dict()
            warm_dict = warm_dict["Value"]

            cold_input = [cold_dict]
            warm_input = [warm_dict]

            temp_df1 = pd.DataFrame(np.array(cold_input), columns=[j])
            temp_df2 = pd.DataFrame(np.array(warm_input), columns=[j])
            # instead of appending I can edit the value of the specific cell with the input
            # (ah that gets difficult with dictionaries)
            # df_input.at[i, j] = [cold_dict]
            # df_input.iloc[i+len(EPdf), df_input.columns.get_loc(j)] = warm_dict
            # could do some sort of join on the
            df_input = df_input.append(temp_df1, ignore_index=True)
            df_input = df_input.append(temp_df2, ignore_index=True)

    return df_input
