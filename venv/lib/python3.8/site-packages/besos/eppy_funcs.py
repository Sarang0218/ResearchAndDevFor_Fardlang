"""
Functions to interact with eppy and geomeppy.
Some functions also support equivalent operations for json buildings.
"""

# Python Core Imports
import contextlib
import json
import os
import random
import re
import sqlite3
import string
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

# External Libraries
import eppy
import numpy as np
import pandas as pd
from eppy.bunch_subclass import BadEPFieldError
from shapely.geometry import LineString, Point, Polygon

# BESOS Imports
from besos import config
from besos import eplus_funcs as eplus
from besos.IDF_class import IDF, view_epJSON
from besos.errors import ModeError


def view_building(building):
    """Displays a matplotlib 3d plot of the building"""
    if get_mode(building) == "json":
        return view_epJSON(building)
    elif get_mode(building) == "idf":
        return building.view_model()


def merge_building(building_from, building_to):
    """Copy objects in building_from to building_to."""
    mode_from = get_mode(building_from)
    mode_to = get_mode(building_to)
    if mode_to == "idf" and mode_from == "idf":
        for attr in building_from.idfobjects:
            for idfobject in building_from.idfobjects[attr]:
                building_to.copyidfobject(idfobject)
    elif mode_to == "json" and mode_from == "json":
        for attr in building_from:
            if attr in building_to:
                building_to[attr].update(building_from[attr])
            else:
                building_to[attr] = building_from[attr]
    else:
        raise NotImplementedError


def convert_format(s: str, place, mode):
    """Converts energyPlus names to their Eppy equivalents

    :param s: name to convert
    :param place: the type of place that is being used 'field' or 'class'
    :param mode: whether to convert to idf or json formatting
    :return: the converted name
    """
    # TODO: Find an authoritative source for the naming conventions
    if mode == "idf":
        if place == "field":
            return s.replace(" ", "_").replace("-", "")
        if place == "class":
            return s.upper()
    if mode == "json":
        if place == "field":
            return s.replace(" ", "_").replace("-", "_").lower()
        if place == "class":
            # uses camel case, s.title() removes some of the capitals and does not work
            return s
    raise ModeError(message=f"no format defined for place:{place} and mode:{mode}")


def get_mode(building) -> str:
    """Determines whether building uses an idf or json format

    :param building: a building, either an IDF or a dictionary of json data
    :return: "idf" or "json"
    """
    if isinstance(building, IDF):
        return "idf"
    if isinstance(building, dict):
        return "json"
    raise ModeError(message=f"Cannot find a valid mode for {building}")


def get_idf(
    idf_file: str = config.files.get("idf"),
    idd_file: str = None,
    version=None,
    output_directory=config.out_dir,
    ep_path=None,
) -> IDF:
    """Uses eppy to read an idf file and generate the corresponding idf object"""
    if version is None:
        version = eplus.get_idf_version_pth(idf_file)
    if idd_file is None:
        # set iddname to default value if needed.
        _, ep_directory = eplus.get_ep_path(version, ep_path)
        default_idd = Path(ep_directory, "Energy+.idd").resolve()
        if IDF.iddname is None:
            IDF.setiddname(str(default_idd))
        elif Path(IDF.iddname).resolve() != default_idd:
            warnings.warn(
                f"idd is already set to: {IDF.iddname}. "
                f"It will NOT be changed to the default of: {default_idd}."
            )
    else:
        # Trying to change the idd file inside a program will cause an error
        # paths must be converted to strings before sending to eppy
        IDF.setiddname(str(idd_file))
    # TODO: Fix this rather than hiding it.
    # calling IDF causes a warning to appear, currently redirect_stdout hides this.
    with contextlib.redirect_stdout(None):
        idf = IDF(idf_file)
    # override the output location so I stop messing up
    idf.run = partial(idf.run, output_directory=output_directory)
    return idf


def get_building(
    building=None,
    data_dict=None,
    output_directory=config.out_dir,
    mode=None,
    version=None,
    ep_path=None,
):
    """Get building from directory"""
    if mode is None and building is None:
        building = config.files.get("building")
        version = eplus.get_idf_version_pth(building)
        mode = config.energy_plus_mode
    if mode is None:
        ending = Path(building).suffix
        if ending == ".idf":
            mode = "idf"
        elif ending == ".epJSON":
            mode = "json"
        else:
            raise ModeError(ending)
    if building is None:
        building = config.files[mode]
    if version is None:
        version = eplus.get_idf_version_pth(building)
    if mode == "idf":
        idf = get_idf(
            idf_file=building,
            version=version,
            idd_file=data_dict,
            output_directory=output_directory,
            ep_path=ep_path,
        )
        return idf
    elif mode == "json":
        if data_dict is not None:
            warnings.warn(
                "epJSON does not support data_dict in get_building, pass into run_building if running EnergyPlus with 'schema_file' argument."
            )
        with open(building) as f:
            return json.load(f)
    raise ModeError(mode)


def get_idfobject_from_name(
    idf: IDF, name: str
) -> Optional[eppy.bunch_subclass.EpBunch]:
    """Gets an object from the passed idf where it's name
    value is equal to the passed string, if none are found
    then this method returns None


    :param idf: The idf to find the object from.
    :param name: The string to find that is equal to the name field of the object.
    :return: the object from the idf with the matching name.
    """

    #: Scan all objects in all lists
    for idfobject_list in idf.idfobjects.values():
        for obj in idfobject_list:
            #: Try except block incase object does not
            #: have a .Name attribute
            try:
                # If the object has the name we are looking for return it.
                if obj.Name == name:
                    return obj
            # ignore objects with no Name
            # Note that the BadEPFieldError may not be possible
            except (BadEPFieldError,):
                pass
    #: If this is encountered then no match was found, so we return None.
    return None


def get_windows(building):
    mode = get_mode(building)
    if mode == "idf":
        return (
            (window.Name, window)
            for window in building.idfobjects["FENESTRATIONSURFACE:DETAILED"]
        )
    elif mode == "json":
        return building["FenestrationSurface:Detailed"].items()
    else:
        raise ModeError(mode)


def wwr_all(building, wwr: float, direction=None) -> None:
    """Sets the wwr for all walls that have a window.
    Will malfunction if there are multiple windows on one wall
    """
    mode = get_mode(building)
    windows = get_windows(building)
    if direction is None:
        for window_name, window in windows:
            if "door" in window_name:
                warnings.warn(
                    f"\nGlass door was detected: {window_name}.\n"
                    f"Door will be resized with wwr_all\n"
                )
            wwr_single(window, wwr, building, mode)
    else:
        values = get_window_range(windows, building, mode)
        # TODO called get_windows twice, hard coded solution
        windows = get_windows(building)
        for window_name, window in windows:
            warnings.warn(
                f"\nGlass door was detected: {window_name}.\n"
                f"Door will be resized with wwr_all\n"
            )
            wwr_single(window, wwr, building, mode, direction, values)


def get_window_range(windows, building, mode):
    """get max and min coordinates range of windows"""
    values = {"max_x": -999, "min_x": 999, "max_y": -999, "min_y": 999}
    for window_name, window in windows:
        # getting coordinates here
        coordinates = get_coordinates(window, building, mode)
        xs = coordinates["xs"]
        ys = coordinates["ys"]
        zs = coordinates["zs"]
        # store max and min values so that we can know the range of coordinates
        if max(xs) > values["max_x"]:
            values["max_x"] = max(xs)
        if min(xs) < values["min_x"]:
            values["min_x"] = min(xs)
        if max(ys) > values["max_y"]:
            values["max_y"] = max(ys)
        if min(ys) < values["min_y"]:
            values["min_y"] = min(ys)
    return values


def get_coordinates(window, building, mode):
    """Get coordinates of the xs, ys, and zs"""
    if mode == "idf":

        def coordinates(ax):
            return [window[f"Vertex_{n}_{ax.upper()}coordinate"] for n in range(1, 5)]

    elif mode == "json":

        def coordinates(ax):
            return [window[f"vertex_{n}_{ax.lower()}_coordinate"] for n in range(1, 5)]

    else:
        raise ModeError(mode)
    coordinate = {}
    coordinate.update(
        {"xs": coordinates("X"), "ys": coordinates("Y"), "zs": coordinates("Z")}
    )
    return coordinate


def set_vertex(idfObj, vertexNum: int, x: float = 0, y: float = 0, z: float = 0):
    """Sets a single vertex of the passed idfObject (idfObj)
    to the specified x,y and z coordinates."""
    for val, name in zip((x, y, z), "XYZ"):
        idfObj["Vertex_{}_{}coordinate".format(vertexNum, name)] = round(val, 2)


def one_window(building):
    """Removes some windows so that each wall has no more than one"""
    mode = get_mode(building)
    walls = set()
    to_remove = []
    windows = get_windows(building)
    for window_name, window in windows:
        if mode == "idf":
            wall_name = window.Building_Surface_Name
        elif mode == "json":
            wall_name = window["building_surface_name"]
        else:
            raise ModeError(mode)
        if wall_name in walls:
            to_remove.append((window_name, window))
        else:
            walls.add(wall_name)
    if mode == "idf":
        for window_name, window in to_remove:
            building.idfobjects["FENESTRATIONSURFACE:DETAILED"].remove(window)
    elif mode == "json":
        for window_name, window in to_remove:
            building["FenestrationSurface:Detailed"].pop(window_name)
    else:
        raise ModeError(mode)


def wwr_single(window, wwr: float, building, mode, direction=None, values=None):
    """Process a single `window` to have the window to wall ratio specified by `wwr`"""

    # will not work for some orientations
    # multiple windows on a single wall will break this
    coordinates = get_coordinates(window, building, mode)
    # getting coordinates here
    xs = coordinates["xs"]
    ys = coordinates["ys"]
    zs = coordinates["zs"]
    # check the alignments
    if max(ys) == min(ys):
        axis = "x"
        axs = xs
    elif max(xs) == min(xs):
        axis = "y"
        axs = ys
    else:
        raise ValueError("The window is not aligned with the x or y axes")
    # with direction specified or not
    if direction is not None:
        if check_direct(direction, xs, ys, values):
            set_wwr_single(window, wwr, axs, zs, axis)
    else:
        set_wwr_single(window, wwr, axs, zs, axis)


# TODO document xs and ys
def check_direct(direction: str, xs, ys, values) -> bool:
    """check if the window's direction is the same as the desired direction

    :param direction: the direction to compare with
    :param values: the dictionary of max and min range of all windows
    :return:
    """
    direction = direction.upper()
    if direction not in {"NORTH", "WEST", "EAST", "SOUTH"}:
        raise NameError("Direction should be either north, east, west, or south")
    if type(direction) != str:
        raise TypeError("Direction should be a string")
    direct = ""
    if max(ys) == min(ys):
        if max(ys) == values["max_y"]:
            direct = "NORTH"
        else:
            direct = "SOUTH"
    elif max(xs) == min(xs):
        if max(xs) == values["max_x"]:
            direct = "EAST"
        else:
            direct = "WEST"
    return direct == direction


# TODO: Document axis
def set_wwr_single(window, wwr: float, axs, zs, axis):
    """Set the single window's wwr

    :param window: the window to be modified
    :param wwr: the window to wall ratio of the result
    :param axs: the axis this window is aligned with
    :param zs: the z coordinate of the window
    """
    width = max(axs) - min(axs)
    scale_factor = np.sqrt(wwr)
    new_width = width * scale_factor
    height = max(zs) - min(zs)
    new_height = height * scale_factor

    start_width = (width - new_width) / 2
    end_width = start_width + new_width
    start_height = (height - new_height) / 2
    end_height = start_height + new_height
    # Maintains vertex order by mimicking the current order
    s = [0] * 4
    for vertex in range(0, 4):
        if zs[vertex] == max(zs):
            # vertex on the top
            if axs[vertex] == max(axs):
                # TOP RIGHT
                s[0] += 1
                set_vertex(window, vertex + 1, z=end_height, **{axis: end_width})
            else:
                # TOP LEFT
                s[1] += 1
                set_vertex(window, vertex + 1, z=end_height, **{axis: start_width})
        else:
            if axs[vertex] == max(axs):
                # BOTTOM RIGHT
                s[2] += 1
                set_vertex(window, vertex + 1, z=start_height, **{axis: end_width})
            else:
                # BOTTOM LEFT
                s[3] += 1
                set_vertex(window, vertex + 1, z=start_height, **{axis: start_width})
    assert s == [1] * 4, ("verticesS are wrong:", s)


def set_daylight_control(building, zone_name, distance, illuminance=500):
    """Set daylighting control to the biggest window of the zone.

    :param building: an idf object
    :param zone_name: name of the zone
    :param distance: the distance from the reference point to the window
    :param illuminance: illuminance setpoint at reference point
    """
    surfs = [
        s.Name
        for s in building.idfobjects["BUILDINGSURFACE:DETAILED"]
        if s.Surface_Type.upper() == "WALL" and s.Zone_Name == zone_name
    ]
    windows = [
        w
        for w in building.idfobjects["FENESTRATIONSURFACE:DETAILED"]
        if w.Surface_Type.upper() == "WINDOW" and w.Building_Surface_Name in surfs
    ]
    if not windows:
        raise ValueError(f"No window found in {zone_name}.")
    window = windows[0]
    max_area = window.area
    for w in windows[1:]:
        area = w.area
        if area > max_area:
            max_area = area
            window = w
    vertex = get_vertices(building, zone_name, window, distance)
    if vertex is None:
        raise ValueError(f"Unable to find a daylighting reference point.")
    building.newidfobject(
        "Daylighting:Controls".upper(),
        Name=f"{zone_name} Daylighting Control",
        Zone_Name=zone_name,
        Minimum_Input_Power_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control=0.01,
        Minimum_Light_Output_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control=0.01,
        Glare_Calculation_Daylighting_Reference_Point_Name=f"{zone_name} Daylighting Reference Point",
        Daylighting_Reference_Point_1_Name=f"{zone_name} Daylighting Reference Point",
        Illuminance_Setpoint_at_Reference_Point_1=illuminance,
    )
    building.newidfobject(
        "Daylighting:ReferencePoint".upper(),
        Name=f"{zone_name} Daylighting Reference Point",
        Zone_Name=zone_name,
        XCoordinate_of_Reference_Point=vertex[0],
        YCoordinate_of_Reference_Point=vertex[1],
        ZCoordinate_of_Reference_Point=vertex[2],
    )


def get_vertices(building, zone_name, window, distance):
    v1 = (
        window.Vertex_1_Xcoordinate,
        window.Vertex_1_Ycoordinate,
        window.Vertex_1_Zcoordinate,
    )
    v2 = (
        window.Vertex_2_Xcoordinate,
        window.Vertex_2_Ycoordinate,
        window.Vertex_2_Zcoordinate,
    )
    v3 = (
        window.Vertex_3_Xcoordinate,
        window.Vertex_3_Ycoordinate,
        window.Vertex_3_Zcoordinate,
    )
    vertices = [v1, v2, v3]

    def subtract(v1, v2):
        return v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]

    for i in range(3):
        for j in range(i + 1, 3):
            line = subtract(vertices[i], vertices[j])
            if line[2] == 0:
                point1, midpoint = (
                    vertices[j],
                    (
                        (vertices[j][0] + vertices[i][0]) / 2,
                        (vertices[j][1] + vertices[i][1]) / 2,
                    ),
                )
            elif line[1] == 0 or line[0] == 0:
                height = abs(line[2]) / 2
    r_point1, r_point2 = get_reference_point(point1[:2], midpoint, distance)
    r_point = point_in_zone(building, zone_name, [r_point1, r_point2])
    if r_point is not None:
        return r_point + (height,)
    return None


def point_in_zone(building, zone_name, points):
    surfs = building.idfobjects["BUILDINGSURFACE:DETAILED"]
    zone_surfs = [s for s in surfs if s.Zone_Name == zone_name]
    floors = [s for s in zone_surfs if s.Surface_Type.upper() == "FLOORS"]
    for floor in floors:
        for p in points:
            if point_in_surface(floor, p):
                return p
    roofs = [s for s in zone_surfs if s.Surface_Type.upper() in ["ROOF", "CEILING"]]
    for roof in roofs:
        for p in points:
            if point_in_surface(roof, p):
                return p
    return None


def point_in_surface(surface, p):
    v1 = (surface.Vertex_1_Xcoordinate, surface.Vertex_1_Ycoordinate)
    v2 = (surface.Vertex_2_Xcoordinate, surface.Vertex_2_Ycoordinate)
    v3 = (surface.Vertex_3_Xcoordinate, surface.Vertex_3_Ycoordinate)
    v4 = (surface.Vertex_4_Xcoordinate, surface.Vertex_4_Ycoordinate)
    poly = Polygon((v1, v2, v3, v4))
    point = Point(p)
    return poly.contains(point)


def get_reference_point(p1, p2, distance):
    line1 = LineString([p1, p2])
    left = line1.parallel_offset(distance, "left")
    right = line1.parallel_offset(distance, "right")
    p3 = left.boundary[1]
    p4 = right.boundary[0]
    return (p3.x, p3.y), (p4.x, p4.y)


def read_sql(path: str, cmds: list, direction=None):
    """Open sql file with connect

    :param path: absolute directory to the sql file.
    :param cmds: list of commands need to be processed, 'all' will process all cmds.
        All commands available : 'all', 'wall area', 'ceiling height', 'floor area', 'volume'
    :param direction: a list of directions of walls when process wall area cmd,
        None means taking both area and gross area of walls in all directions and
        the total areas of all walls.
    :return: a dictionary of all data desired.

    """
    try:
        sql_path = path
        conn = sqlite3.connect(sql_path)
        cur = conn.cursor()
    except IOError:
        print("Either no such file or file path is not correct.")

    if not isinstance(cmds, list):
        raise TypeError("cmds has be a list")

    def get_azim_range(direction: str):
        if direction == "total":
            azim_range = [0, 360]
        elif direction == "north":
            azim_range = [0, 44.99, 315, 360]
        elif direction == "east":
            azim_range = [45, 135]
        elif direction == "south":
            azim_range = [135, 269.99]
        elif direction == "west":
            azim_range = [270, 515]
        else:
            raise NameError(
                "Wrong direction! attention on the str, 'north', 'south', 'west', 'east'"
            )
        return azim_range

    def wall_area(dire: str):
        """Find total area with window subtracted from either all walls or only walls in current direction"""
        table_name = "Surfaces"
        azim_range = get_azim_range(dire)
        if len(azim_range) > 2:
            query = (
                f"SELECT SUM(Area), SUM(GrossArea) FROM {table_name} WHERE "
                f"(ClassName = 'Wall' AND ((Azimuth >= {azim_range[0]} AND "
                f"Azimuth <= {azim_range[1]}) or (Azimuth >= {azim_range[2]} "
                f"AND Azimuth <= {azim_range[3]})))"
            )
        else:
            query = (
                f"SELECT SUM(Area), SUM(GrossArea) FROM {table_name} WHERE "
                f"(ClassName = 'Wall' AND Azimuth >= {azim_range[0]} AND "
                f"Azimuth <= {azim_range[1]})"
            )
        cur.execute(query)
        data = cur.fetchall()
        return data

    def ceiling_height():
        """Find height of the ceiling"""
        floor = floor_area()
        volume = zone_volume()
        height = volume / floor
        return height

    def floor_area():
        """Find total floor area"""
        table_name = "Zones"
        query = f"SELECT SUM(FloorArea) FROM {table_name}"
        cur.execute(query)
        data = cur.fetchall()
        return data[0][0]

    def zone_volume():
        """Find total volume of Zones"""
        table_name = "Zones"
        query = f"SELECT SUM(Volume) FROM {table_name}"
        cur.execute(query)
        data = cur.fetchall()
        return data[0][0]

    def sg_temp():
        """find site ground temperature"""
        table_name = "ReportDataDictionary"
        query = f"SELECT ReportDataDictionaryIndex FROM {table_name} WHERE Name = 'Site Ground Temperature'"
        cur.execute(query)
        index = cur.fetchall()
        table_name = "ReportData"
        query = f"SELECT Value FROM {table_name} WHERE ReportDataDictionaryIndex = {index[0][0]}"
        cur.execute(query)
        data = cur.fetchall()
        return data

    # Taking commands, there is a small problem is that the code is not checking if the direction is correct.
    # And there is no protection when using cmd all and others
    def pull_data():
        result = {}
        # decide which direction we want
        if direction is None:
            direct = ["total", "south", "north", "west", "east"]
        else:
            direct = direction
        if len(cmds) == 1 and cmds[0] == "all":
            result.update(
                {
                    "ceiling height": ceiling_height(),
                    "floor area": floor_area(),
                    "volume": zone_volume(),
                }
            )
            for dire in direct:
                area = wall_area(dire)
                result.update(
                    {
                        f"{dire} wall area": area[0][0],
                        f"{dire} wall gross area": area[0][1],
                    }
                )
        else:
            for cmd in cmds:
                if cmd == "all":
                    raise NameError(
                        "There should not be an 'all' command with other commands"
                    )
                elif cmd == "wall area":
                    for dire in direct:
                        area = wall_area(dire)
                        result.update(
                            {
                                f"{dire} wall area": area[0][0],
                                f"{dire} wall gross area": area[0][1],
                            }
                        )
                elif cmd == "ceiling height":
                    result.update({"ceiling height": ceiling_height()})
                elif cmd == "floor area":
                    result.update({"floor area": floor_area()})
                elif cmd == "volume":
                    result.update({"volume": zone_volume()})
                elif cmd == "SGtemp":
                    result.update({"site ground temp": sg_temp()})
                else:
                    raise NameError(
                        "No such command, available cmds are 'all', 'wall area', "
                        "'ceiling height', 'floor area', 'volume', 'SGtemp'"
                    )
        return result

    result = pull_data()
    conn.close()
    return result


# TODO make this function customizable for more cmds, not just a handler for one task.
# shape option is for special use, might remove it in the future
# number of floor is also a temp solution
# Add direction
# Add cmds
def write_csv(path: str, dest: str, shape=False):
    """A handler to call read_sql() function and get data, then put the data into exl file

    :param path: path to read sql file
    :param dest: destination to put data
    :param shape: for special use, only works with right file name
    """
    # Get data from sql file. Currently retrieves all the data.
    data = read_sql(
        path,
        ["all"],
    )

    # Get number of floors and shape from file name
    filename = Path(dest).name
    match = re.search(r"M.idf", filename)
    if match is None:
        num_of_floor = 1
    else:
        num_of_floor = 3
    if shape:
        shape = filename[0].upper()

    # Read csv file
    df = pd.read_csv(dest)
    wall_area = []
    south_wall_area = []
    total_rows = len(df["Window to Wall Ratio"])
    for i in range(total_rows):
        total_area = data["total wall gross area"] * (1 - df["Window to Wall Ratio"][i])
        south_area = data["south wall gross area"] * (1 - df["Window to Wall Ratio"][i])
        wall_area.append(total_area)
        south_wall_area.append(south_area)

    # put data back to csv
    df["wall area net"] = wall_area
    df["wall area gross"] = data["total wall gross area"]
    df["south wall area net"] = south_wall_area
    df["south wall area gross"] = data["south wall gross area"]
    df["number of floors"] = num_of_floor
    df["ceiling height"] = data["ceiling height"]
    df["total floor area"] = data["floor area"]
    df["total volume"] = data["volume"]
    if shape:
        df["shape"] = shape
    df.to_csv(dest)


def generate_dir(dest_folder=None):
    """func use to generate a directory for besos outputs"""
    folder = "BESOS_Output"
    if dest_folder is not None:
        folder = dest_folder
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=20))
    while os.path.exists(Path(folder, res)):
        res = "".join(random.choices(string.ascii_uppercase + string.digits, k=20))
    os.makedirs(Path(folder, res))
    dir_ = Path(folder, res)
    return dir_


def generate_batch(
    account: str,
    time: str,
    email: str,
    task_id: int,
    cpu_per_task=1,
    mem=1000,
):
    """function to write a bash file for running jupyter on computer canada

    :param account: account used
    :param time: time for bash job
    :param email: user email
    :param task_id: task id to use
    :param cpu_per_task: number of cpu used to run one task
    :param mem: memory needs in total
    """
    # TODO there might be a calculation relationship between mem, cpu and task_id.
    # If possible give some default value to the arguments

    f = open("clusterbatch.sh", "w")
    f.write(
        f"#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        f"\n"
        f"#SBATCH --array=1-250\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --cpus-per-task={cpu_per_task}\n"
        f"#SBATCH --mem={mem}mb\n"
        f"#SBATCH --mail-user={email}\n"
        f"\n"
        f"#!generate the virtual environment\n"
        f"module load python/3.6\n"
        f"source ~/env/bin/activate\n"
        f"echo 'program started at:`date`'\n"
        f"srun python cluster.py $SLURM_ARRAY_TASK_ID {task_id}\n"
        f"deactivate\n"
        f"echo 'program ended at: `date`'\n"
    )
    f.close()


def convert_to_json(idf: IDF):
    """convert idf file to json
    the func will create a json file that is converted from the idf
    can use --convert-only when energyplus version 9.3 is in besos
    """
    os.system(f"energyplus -c {idf}")
