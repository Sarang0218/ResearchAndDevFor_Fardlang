"""Enables some geomeppy based functionality on idf objects"""

# External Libraries
from geomeppy import view_geometry, recipes, IDF as geomeppy_idf
from numpy import sin, cos, deg2rad
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError):
    # this isn't always needed so we can ignore if it's not present
    pass


class IDF(geomeppy_idf):
    """Adding some class functions to help with Geomeppy"""

    def read(self):
        """don't need to use allcaps to search for items in idfobjects."""

        class AllCapsDict(dict):
            def __getitem__(self, key):
                return super().__getitem__(key.upper())

        super().read()
        self.idfobjects = AllCapsDict(self.idfobjects)

    def relative_to_world_coords(self):
        """translates all surfaces, subsurfaces, shadings and daylighting reference points
        to world coordinates"""
        relative_to_world_coords(self)

    def remove_windows(self, orientation=None):
        """removes windows with respect to orientation
        Note: default is ALL directions
        """
        remove_windows(self, orientation)

    def add_overhangs(self, depth, tilt=90, orientation=None, transmittance=""):
        """Set the overhang shading on all external windows by creating a SHADING:ZONE:DETAILED object
        which can be viewed using geomeppy's view_model() function.
        Note:
        Depth must be greater than 0
        :param depth: The depth of the overhang {m}.
        :param tilt: Tilt Angle from Window/Door {deg}.
        :param orientation: One of "north", "east", "south", "west". Walls within 45 degrees will be affected.
        :param Transmittance: Transmittance Schedule Name.
        """
        add_overhangs(self, depth, tilt, orientation, transmittance)

    def remove_shading(self, orientation=None):
        """removes shading surfaces with respect to orientation
        Note: default is ALL directions
        """
        remove_shading(self, orientation)

    # def translate_to_origin(self):
    #     """Translates building so it is touching the origin
    #     Note: overrides Geomeppy's translate_to_origin()
    #     """
    #     translate_to_origin(self)

    def view_model(self):
        """automatically checks if the idf is in Relative or World coordinates"""
        ggr = self.idfobjects["GLOBALGEOMETRYRULES"][0]
        if ggr.Coordinate_System == "Relative":
            return relative_coordinates_model(self)
        return view_idf(self)


def view_idf(idf):
    # type: (IDF) -> None
    """Display an IDF for inspection."""
    fig = plt.figure()

    # create the figure and add the surfaces
    ax = fig.add_subplot(projection="3d")
    collections = view_geometry._get_collections(idf, opacity=0.5)
    for c in collections:
        ax.add_collection3d(c)

    # calculate and set the axis limits
    limits = view_geometry._get_limits(idf=idf)
    ax.set_xlim(limits["x"])
    ax.set_ylim(limits["y"])
    ax.set_zlim(limits["z"])

    fig = plt.gcf()
    return fig, ax


def remove_shading(idf, orientation=None):

    """removes fins and overhangs from the idf
    Note:
    cannot differentiate between fins and overhangs.

    :param: orientation: compass direction with respect to building azimuth

    """

    from eppy.bunch_subclass import BadEPFieldError

    # orientation to degrees
    orientations = {
        "north": 0.0,
        "east": 90.0,
        "south": 180.0,
        "west": 270.0,
        None: None,
    }

    degrees = orientations.get(orientation, None)
    external_walls = filter(
        lambda x: x.Outside_Boundary_Condition.lower() == "outdoors",
        idf.getsurfaces("wall"),
    )

    external_walls = list(
        filter(lambda x: recipes._has_correct_orientation(x, degrees), external_walls)
    )
    windows = idf.getsubsurfaces("window")
    windows_walls = []
    for wall in external_walls:
        for window in windows:
            if window.Building_Surface_Name == wall.Name:
                windows_walls.append(window)
                windows_walls.append(wall)

    shading = idf.getshadingsurfaces()

    to_remove = [
        "Shading:Site:Detailed",
        "Shading:Building:Detailed",
    ]  # objects mentioned do not contain fins and overhangs
    for shade in list(shading):
        if shade.obj[0] in to_remove:
            shading.remove(shade)

    for surface in windows_walls:
        for shade in list(shading):

            try:
                if shade.Base_Surface_Name == surface.Name:
                    idf.removeidfobject(shade)
                    shading.remove(shade)
            except BadEPFieldError:
                if shade.Window_or_Door_Name == surface.Name:
                    idf.removeidfobject(shade)
                    shading.remove(shade)
            except BadEPFieldError:
                raise BadEPFieldError(
                    "unable to find field Base_Surface_Name and Window_or_Door_Name"
                )


def relative_to_world_coords(idf):

    """translates all surfaces, subsurfaces and daylighting reference points to world coordinates and
    then sets global geometry rules coordinate systems to "World"
    """
    # requires geomeppy.recipes.translate_coords
    from tqdm.notebook import tqdm

    shading = idf.idfobjects["SHADING:ZONE:DETAILED"]
    windows = idf.getsubsurfaces("window")
    surfaces = idf.getsurfaces()
    zones = idf.idfobjects["ZONE"]
    daylighting_refs = idf.idfobjects["DAYLIGHTING:REFERENCEPOINT"]

    for zone in tqdm(zones, desc="Relative to World Coordinates"):
        x, y, z = zone.X_Origin, zone.Y_Origin, zone.Z_Origin
        if zone.Direction_of_Relative_North == 180:
            x, y, z = -zone.X_Origin, -zone.Y_Origin, zone.Z_Origin
        zone_coords = (x, y, z)

        for surface in surfaces:
            coords = surface.coords
            if surface.Zone_Name == zone.Name:
                new_coords = recipes.translate_coords(coords, zone_coords)

                for window in windows:
                    if window.Building_Surface_Name == surface.Name:
                        win_coords = recipes.translate_coords(
                            window.coords, zone_coords
                        )
                        window.setcoords(win_coords)

                for shade in shading:
                    if shade.Base_Surface_Name == surface.Name:
                        shade_coords = recipes.translate_coords(
                            shade.coords, zone_coords
                        )
                        shade.setcoords(shade_coords)
                surface.setcoords(new_coords)

        for point in daylighting_refs:
            if point.Zone_Name == zone.Name:
                point.XCoordinate_of_Reference_Point += zone.X_Origin
                point.YCoordinate_of_Reference_Point += zone.Y_Origin
                point.ZCoordinate_of_Reference_Point += zone.Z_Origin

    ggr = idf.idfobjects["GLOBALGEOMETRYRULES"][0]
    ggr.Coordinate_System = "World"
    ggr.Daylighting_Reference_Point_Coordinate_System = "World"
    ggr.Rectangular_Surface_Coordinate_System = "World"


def add_overhangs(idf, depth=float, tilt=90, orientation=None, Transmittance=""):
    # uses geomeppy.recipes._has_correct_orientation

    """Set the overhang shading on all external windows by creating a SHADING:ZONE:DETAILED object
        which can be viewed using geomeppy's view_model() function.

    Note:
    Depth must be greater than 0

    :param idf: The IDF to edit.
    :param depth: The depth of the overhang {m}.
    :param tilt: Tilt Angle from Window/Door {deg}.
    :param orientation: One of "north", "east", "south", "west". Walls within 45 degrees will be affected.
    :param Transmittance: Transmittance Schedule Name.

    """

    try:
        ggr = idf.idfobjects["GLOBALGEOMETRYRULES"][0]  # type: Optional[Idf_MSequence]
    except IndexError:
        ggr = None
    # orientation to degrees
    orientations = {
        "north": 0.0,
        "east": 90.0,
        "south": 180.0,
        "west": 270.0,
        None: None,
    }
    degrees = orientations.get(orientation, None)
    external_walls = filter(
        lambda x: x.Outside_Boundary_Condition.lower() == "outdoors",
        idf.getsurfaces("wall"),
    )
    external_walls = list(
        filter(lambda x: recipes._has_correct_orientation(x, degrees), external_walls)
    )
    windows = idf.getsubsurfaces("window")
    for wall in external_walls:

        for window in windows:

            if window.Building_Surface_Name == wall.Name:
                coords = [window.coords[0]]
                x, y, z = window.coords[1]
                coords.append(
                    (
                        x + depth * sin(deg2rad(window.azimuth)) * sin(deg2rad(tilt)),
                        y + depth * cos(deg2rad(window.azimuth)) * sin(deg2rad(tilt)),
                        window.coords[0][2] - depth * cos(deg2rad(tilt)),
                    )
                )
                x, y, z = window.coords[2]
                coords.append(
                    (
                        x + depth * sin(deg2rad(window.azimuth)) * sin(deg2rad(tilt)),
                        y + depth * cos(deg2rad(window.azimuth)) * sin(deg2rad(tilt)),
                        window.coords[3][2] - depth * cos(deg2rad(tilt)),
                    )
                )
                coords.append(window.coords[3])
                Shade = idf.newidfobject(
                    "SHADING:ZONE:DETAILED",
                    Name="%s - Overhang" % window.Name,
                    Transmittance_Schedule_Name=Transmittance,
                    Base_Surface_Name=window.Building_Surface_Name,
                )
                Shade.setcoords(coords, ggr)


def translate_to_origin(idf):
    # type: (IDF) -> None

    """Move an IDF so the building touches the origin.
    :param idf: The IDF to edit.

    """
    from geomeppy.geom.polygons import Polygon3D

    surfaces = idf.getsurfaces()
    daylighting_refs = idf.idfobjects["DAYLIGHTING:REFERENCEPOINT"]
    zones = idf.idfobjects["ZONE"]

    min_y = min(min(Polygon3D(s.coords).ys) for s in surfaces)
    surfaces = filter(lambda x: min(Polygon3D(x.coords).ys) == min_y, surfaces)
    min_x = min(min(Polygon3D(s.coords).xs) for s in surfaces)

    idf.translate((-min_x, -min_y))

    for point in daylighting_refs:
        point.XCoordinate_of_Reference_Point -= min_x
        point.YCoordinate_of_Reference_Point -= min_y
    for zone in zones:
        if zone.Direction_of_Relative_North == 0:
            zone.X_Origin -= min_x
            zone.Y_Origin -= min_y
        elif zone.Direction_of_Relative_North == 180:
            zone.X_Origin += min_x
            zone.Y_Origin += min_y

    for point in daylighting_refs:
        point.XCoordinate_of_Reference_Point -= min_x
        point.YCoordinate_of_Reference_Point -= min_y
    for zone in zones:
        if zone.Direction_of_Relative_North == 0:
            zone.X_Origin -= min_x
            zone.Y_Origin -= min_y
        elif zone.Direction_of_Relative_North == 180:
            zone.X_Origin += min_x
            zone.Y_Origin += min_y


def relative_coordinates_model(building):
    # uses geomeppy.recipes.translate_coords()

    """translates RELATIVE coordinates of building surfaces to be viewed
        using view_model() function without altering original idf.

    Note:
    Does not correctly display after using geomeppy's rotate() function

    :param: building: The building we want displayed with RELATIVE coordinates

    """
    from copy import deepcopy

    # creates a copy of the building so it doesn't edit original idf
    idf = deepcopy(building)
    relative_to_world_coords(idf)
    idf.translate_to_origin()
    return idf.view_model()


def remove_windows(idf, orientation=None):
    # uses geomeppy.recipes._has_correct_orientation

    """removes windows from the idf

    :param: orientation: compass direction with respect to building north

    """

    # orientation to degrees
    orientations = {
        "north": 0.0,
        "east": 90.0,
        "south": 180.0,
        "west": 270.0,
        None: None,
    }
    degrees = orientations.get(orientation, None)
    external_walls = filter(
        lambda x: x.Outside_Boundary_Condition.lower() == "outdoors",
        idf.getsurfaces("wall"),
    )
    external_walls = list(
        filter(lambda x: recipes._has_correct_orientation(x, degrees), external_walls)
    )
    windows = idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]
    external_wall_names = [wall.Name for wall in external_walls]

    for window in list(windows):
        if window.Building_Surface_Name in external_wall_names:
            idf.removeidfobject(window)


"""Some epJSON model viewing"""


def _get_collections(building, opacity=0.5):
    """Set up 3D collections for each surface type."""
    if isinstance(building, dict):
        surfaces = building.get("BuildingSurface:Detailed", {})
        fenestrations = building.get("FenestrationSurface:Detailed", {})
        shading_surfaces = building.get("Shading:Zone:Detailed", {})
    else:
        surfaces = building["BuildingSurface:Detailed"]
        fenestrations = building["FenestrationSurface:Detailed"]
        shading_surfaces = building["Shading:Zone:Detailed"]
    # set up the collections
    walls = _get_collection("wall", surfaces, opacity, facecolor="lightyellow")
    floors = _get_collection("floor", surfaces, opacity, facecolor="dimgray")
    roofs = _get_collection("roof", surfaces, opacity, facecolor="firebrick")
    shading = _get_collection(
        "shading", shading_surfaces, opacity, facecolor="darkolivegreen"
    )
    windows = _get_windows("window", fenestrations, opacity, facecolor="cornflowerblue")

    return walls, roofs, floors, shading, windows


def _get_collection(surface_type, surfaces, opacity, facecolor, edgecolors="black"):
    """Make collections from a list of EnergyPlus surfaces."""
    coords = []
    for surface in surfaces:
        if surface_type == "shading":
            surface_coords = []
            for v_name in surfaces[surface]["vertices"]:
                vertices = list(point for vertex, point in v_name.items())
                surface_coords.append(tuple(vertices))
            coords.append(surface_coords)
        elif (
            surfaces[surface]["surface_type"]
            and surfaces[surface]["surface_type"].lower() == surface_type.lower()
        ):
            surface_coords = []
            for v_name in surfaces[surface]["vertices"]:
                vertices = list(point for vertex, point in v_name.items())
                surface_coords.append(tuple(vertices))
            coords.append(surface_coords)

    trimmed_coords = [c for c in coords if c]  # dump any empty surfaces
    collection = Poly3DCollection(
        trimmed_coords, alpha=opacity, facecolor=facecolor, edgecolors=edgecolors
    )
    return collection


def _get_windows(surface_type, fenestrations, opacity, facecolor, edgecolors="black"):
    window_coords = []
    for window in fenestrations:
        vertex_1 = list(
            fenestrations[window][key]
            for key in fenestrations[window]
            if "vertex_1" in key
        )
        vertex_2 = list(
            fenestrations[window][key]
            for key in fenestrations[window]
            if "vertex_2" in key
        )
        vertex_3 = list(
            fenestrations[window][key]
            for key in fenestrations[window]
            if "vertex_3" in key
        )
        vertex_4 = list(
            fenestrations[window][key]
            for key in fenestrations[window]
            if "vertex_4" in key
        )
        window_coords.append(
            [tuple(vertex_1), tuple(vertex_2), tuple(vertex_3), tuple(vertex_4)]
        )
    trimmed_coords = [c for c in window_coords if c]  # dump any empty surfaces
    collection = Poly3DCollection(
        trimmed_coords, alpha=opacity, facecolor=facecolor, edgecolors=edgecolors
    )
    return collection


def _get_limits(building):
    if isinstance(building, dict):
        surfaces = building.get("BuildingSurface:Detailed", {})
        surfaces.update(building.get("Shading:Zone:Detailed", {}))
    else:
        surfaces = building["BuildingSurface:Detailed"]
        surfaces.update(building["Shading:Zone:Detailed"])
    x_coords = []
    y_coords = []
    z_coords = []
    for surface in surfaces:
        vertices = surfaces[surface]["vertices"]
        for point in vertices:
            x_coords.append(point["vertex_x_coordinate"])
            y_coords.append(point["vertex_y_coordinate"])
            z_coords.append(point["vertex_z_coordinate"])
    delta = max(
        (max(x_coords) - min(x_coords)),
        (max(y_coords) - min(y_coords)),
        (max(z_coords) - min(z_coords)),
    )
    x_max = min(x_coords) + delta
    y_max = min(y_coords) + delta
    z_max = min(z_coords) + delta
    return {
        "x": (min(x_coords), x_max),
        "y": (min(y_coords), y_max),
        "z": (min(z_coords), z_max),
    }


def view_epJSON(building):
    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    collections = _get_collections(building, opacity=0.5)
    for c in collections:
        ax.add_collection3d(c)

    limits = _get_limits(building)

    ax.set_xlim(limits["x"])
    ax.set_ylim(limits["y"])
    ax.set_zlim(limits["z"])

    return fig, ax
