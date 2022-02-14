"""
Functions related to under-the-hood interactions with energyplus.
"""

# Python Core Libraries
import json
import os
import platform
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

# External Libraries
from deprecated.sphinx import deprecated

# BESOS Imports
from besos import eppy_funcs as ef
from besos import config
from besos import objectives
from besos.errors import ModeError, InstallationError
from besos.besostypes import PathLike


# TODO: Make this a method of the building class when we add it
def get_idf_version(building):
    """Get energyplus version from idf or json building
    :param building: IDF object to find version of.
    :returns: String of version (ex. '9.3.0')
    """
    mode = ef.get_mode(building)
    if mode == "idf":
        return building.idfobjects["VERSION"][0].Version_Identifier
    elif mode == "json":
        try:
            return building["Version"]["Version 1"]["version_identifier"]
        except KeyError:
            try:
                return building["epJSON_schema_version"]
            except:
                raise ModeError(
                    mode="json",
                    message=f"Cannot find IDF version, check epJSON formatting.",
                )
    else:
        raise ModeError(mode)


def get_idf_version_pth(path):
    """Get energyplus version from idf or epJSON path
    :param path: Path to idf to epJSON file.
    :returns: String of version (ex. '9.3.0')
    """
    path = str(path)
    version_lines = (
        ""  # string containing the important lines of the json file to later
    )
    version = ""
    flag = False  # used to save the lines following 'Version' to version_lines
    if path.endswith(".idf"):
        with open(path, "r") as read_obj:  # opening idf file to read
            # Read all lines in the file one by one
            for line in read_obj:
                if flag:
                    version_lines += line.rstrip(
                        "\n"
                    )  # adds the line after 'Version,' is seen in case of line break
                    break
                # For each line, check if line contains 'Version,'
                if ("Version," in line or "VERSION," in line) and not line.startswith(
                    "!"
                ):
                    version_lines += line.rstrip("\n")
                    flag = True
        version = (
            version_lines.split(";")[0].split(",")[-1].replace(" ", "").replace("=", "")
        )
    elif path.endswith(".epJSON"):
        readobj = open(path, "r")
        jsonText = readobj.read()
        readobj.close()
        jsonObj = json.loads(jsonText)
        try:
            version = jsonObj["Version"]["Version 1"]["version_identifier"]
        except KeyError:
            try:
                version = jsonObj["epJSON_schema_version"]
            except KeyError:
                message = f"Version object not found within epJSON file, consider updating epJSON file\nepJSON File: {path}"
                raise ModeError("json", message=message)
    return version


def has_hvac_templates(building) -> bool:
    """Returns whether or not the building contains HVACTemplate objects
    https://bigladdersoftware.com/epx/docs/8-0/input-output-reference/page-061.html

    :param building:
    :return: True if at leas one HVACTemplate object is present in the building.
    """
    mode = ef.get_mode(building)
    prefix = ef.convert_format("HVACTemplate", "class", mode)
    if mode == "idf":
        return any(
            k for k, v in building.idfobjects.items() if k.startswith(prefix) and v
        )
    else:
        return any(k for k in building if k.startswith(prefix))


def run_building(building, out_dir=config.out_dir, version=None, **eplus_args):
    """Run energy plus on a building object and return results
    :param building_path: Path to building file
    :param out_dir: Path to store EnergyPlus output files, if out_dir is not defined the results will not be saved
    :param version: Version of building file
    :param ep_path: Path to EnergyPlus if installed in unexpected directory
    :param expand_objects: Boolean if '--expandobjects' option to be appended to command
    :returns: A dictionary of EnergyPlus outputs
    """
    # backwards compatibility
    if version:
        warnings.warn(
            "the version argument is deprecated for run_building,"
            " and will be removed in the future",
            FutureWarning,
        )
        assert version == get_idf_version(building), "Incorrect version"

    with tempfile.TemporaryDirectory(dir=Path.home(), prefix=".besos_") as temp_dir:
        if out_dir is None:
            out_dir = temp_dir
        try:
            building_path = Path(temp_dir, "in.idf").resolve()
            building.saveas(str(building_path))
        except AttributeError:
            building_path = Path(temp_dir, "in.epJSON").resolve()
            with open(str(building_path), "w") as f:
                json.dump(building, f)
        expand_objects = has_hvac_templates(building)
        run_energyplus(
            building_path,
            out_dir=out_dir,
            version=get_idf_version(building),
            expand_objects=expand_objects,
            **eplus_args,
        )
        return objectives.read_eso(out_dir)


def run_energyplus(
    building_path: PathLike,
    out_dir: PathLike = config.out_dir,
    epw: PathLike = config.files["epw"],
    err_dir: PathLike = config.err_dir,
    schema_file=None,
    error_mode="Silent",
    version=None,
    ep_path=None,
    expand_objects: bool = False,
):
    """Run EnergyPlus. This method is intended to work as similar to the cli tool as possible
    :param building_path: Path to building file
    :param out_dir: Path to store EnergyPlus output files
    :param epw: Path to epw file
    :param err_dir: Path to store EnergyPlus error files
    :param schema_file: Path to data dictionary for EnergyPlus
    :param error_mode: Error mode selection
    :param version: Version of building file
    :param ep_path: Path to EnergyPlus if installed in unexpected directory
    :param expand_objects: Boolean if '--expandobjects' option to be appended
    :returns: None
    """
    if version is None:
        version = get_idf_version_pth(building_path)
    ep_exe_path, ep_directory = get_ep_path(version, ep_path)
    schema_file = schema_file or Path(ep_directory, "Energy+.idd")

    cmd = [
        ep_exe_path,
        "--idd",
        schema_file,
        "--weather",
        epw,
    ]
    if out_dir:
        cmd += ["--output-directory", out_dir]
    if expand_objects:
        cmd.append("--expandobjects")
    cmd.append(building_path)
    needs_shell = platform.system() == "Windows"
    try:
        subprocess.run(cmd, check=True, shell=needs_shell)
    except subprocess.CalledProcessError as e:
        # TODO: This log is excessively noisy. Can we cut it down to just the command's stderr?
        if error_mode != "Silent":
            # print eplus error
            filename = Path(out_dir, "eplusout.err")
            if os.path.exists(filename):
                err_file = open(filename, "r")
                for line in err_file:
                    print(line)
                print()
                err_file.close()
        if err_dir is not None and out_dir != err_dir:
            # copy eplus error files to err_dir
            if os.path.exists(err_dir):
                shutil.rmtree(err_dir)
            shutil.copytree(out_dir, err_dir)
        raise e


def get_ep_path(idf_version, ep_path=None):
    """get EnergyPlus installation path by idf_version
    :param idf_version: idf_version of EnergyPlus
    :param ep_path: Optional - path to EnergyPlus if installed in different location then expected
    :returns: (path to EnergyPlus executable, path to EnergyPlus)
    """
    if len(idf_version) == 3:
        if idf_version == "9.0":
            idf_version = "9-0-1"
        else:
            idf_version = idf_version.replace(".", "-") + "-0"
    else:
        idf_version = idf_version.replace(".", "-")
    if ep_path is not None:
        ep_directory = os.path.abspath(ep_path)
        if not os.path.exists(ep_directory):
            message = f"'ep_path' does not exist.\n'ep_path': {ep_path}"
            raise InstallationError(message=message)
        try:
            cmd = [os.path.join(ep_directory, "energyplus"), "--version"]
            out, err = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()
            ep_path_version = (
                str(out).split("Version ")[-1].split("-")[0].replace(".", "-")
            )
        except:
            message = f"'ep_path' provided does not contain an energyplus file\n'ep_path': {ep_path}"
            raise InstallationError(message=message)
        if ep_path_version != idf_version:
            message = f"'ep_path' provided is a different version of EnergyPlus then 'idf_version' argument\n'ep_path': {ep_path}\n'idf_version': {idf_version}"
            raise InstallationError(message=message)
        if platform.system() == "Windows":
            ep_exe = os.path.join(ep_directory, "energyplus.exe")
        else:
            ep_exe = os.path.join(ep_directory, "energyplus")
    else:
        # this is duplicated from eppy.runner.run_functions.paths_from_version
        if platform.system() == "Windows":
            ep_directory = "C:/EnergyPlusV{idf_version}".format(idf_version=idf_version)
            ep_exe = os.path.join(ep_directory, "energyplus.exe")
        elif platform.system() == "Linux":
            ep_directory = "/usr/local/EnergyPlus-{idf_version}".format(
                idf_version=idf_version
            )
            ep_exe = os.path.join(ep_directory, "energyplus")
        else:
            ep_directory = "/Applications/EnergyPlus-{idf_version}".format(
                idf_version=idf_version
            )
            ep_exe = os.path.join(ep_directory, "energyplus")

    if not os.path.exists(ep_directory) or not os.path.exists(ep_exe):
        raise InstallationError(version=idf_version)
    return ep_exe, ep_directory


def print_available_outputs(
    building,
    version=None,
    name=None,
    frequency=None,
):
    # backwards compatibility
    if version:
        warnings.warn(
            "the version argument is deprecated for print_available_outputs,"
            " and will be removed in the future",
            FutureWarning,
        )
        assert version == get_idf_version(building), "Incorrect version"

    if name is not None:
        name = name.lower()
    if frequency is not None:
        frequency = frequency.lower()
    results = run_building(building)
    for key in results.keys():
        if name is not None:
            if name not in key[0].lower():
                continue
            if frequency is not None and key[1].lower() != frequency:
                continue
        elif frequency is not None:
            if key[1].lower() != frequency:
                continue
        print(list(key))
