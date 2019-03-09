"""
Support to read in exported Petrel surfaces as suitable DataFrames for use as input data.
Currently supported file formats:
    - Earth Vision Grid (ASCII)

@author: Alexander Schaaf
"""
import pandas as pd
import numpy as np


def cut_df(df, e):
    """
    Cuts dataframe of values outside of extent.

    Arguments:
        df (pandas.DataFrame): DataFrame to be filtered along columns x, y, z
        e (list): extent [xmin, xmax, ymin, ymax, ymin, zmax]

    Returns:
        pandas.DataFrame
    """
    return df[(abs(df.X) > abs(e[0])) & (abs(df.X) < abs(e[1])) &
              (abs(df.Y) > abs(e[2])) & (abs(df.Y) < abs(e[3])) &
              (abs(df.Z) > abs(e[4])) & (abs(df.Z) < abs(e[5]))]


def read_earth_vision_grid(fp, formation=None, preserve_colrow=False, group=None, decimate=None):
    """
    Reads Earth Vision Grid files exported by Petrel into GemPy Interfaces-compatible DataFrame.

    Args:
        fp (str): Filepath, e.g. "/surfaces/layer1"
        formation (str): Formation name, Default None
        preserve_colrow (bool): If True preserves row and column values saved in the Earth Vision Grid file. Default False
        group (str): If given creates columns with a group name (useful to later identification of formation subsets). Default None

    Returns:
        pandas.DataFrame
    """
    surface = pd.read_csv(fp, skiprows=20, header=None, delim_whitespace=True)
    surface.columns = "X Y Z col row".split()

    if not formation:
        formation = fp.split("/")[-1]  # take filename

    surface["formation"] = formation

    if not preserve_colrow:
        surface.drop('col', axis=1, inplace=True)
        surface.drop('row', axis=1, inplace=True)

    if group:
        surface["group"] = group

    return surface


def read_list(fps, formations=None, groups=None, preserve_colrow=False):
    """
    Given a list of file paths for EarthVision files, reads them all in and returns gempy interfaces dataframe with all
    points in given EarthVision surfaces.

    Args:
        fps (list): List of filepath strings to EarthVision files
        formations (list): List of corresponding formation names to apply to each read surface.
        groups (list): List of group names (relevant for proper identification of seperate parts of same layer)
        preserve_colrow (bool): If to preserve the col/row parameters stored in EarthVision files, default: False

    Returns:
        pandas.DataFrame: gempy interfaces dataframe
    """
    if preserve_colrow:
        columns = "X Y Z col row".split()
    else:
        columns = "X Y Z".split()
    surfaces = pd.DataFrame(columns=columns)

    for i, fp in enumerate(fps):
        if formations:
            fmt = formations[i]
        else:
            fmt = None

        if groups:
            grp = groups[i]
        else:
            grp = None

        surfaces = surfaces.append(read_earth_vision_grid(fp, formation=fmt, group=grp,
                                                          preserve_colrow=preserve_colrow))

    # if decimate:
    #     if groups:
    #         g = "group"
    #     else:
    #         g = "formation"
    #     surfaces = surfaces.groupby(g).apply(lambda x: x.sample(decimate)).reset_index(drop=True)

    return surfaces


def orientations_by_group(interfaces, group="formation"):
    """
    Calculates one orientation per group in interfaces dataframe, returns orientations dataframe.

    Args:
        interfaces (pandas.DataFrame):
        group (str): Specifies by which column interfaces will be grouped by to calculate orientaitons. Default: "formation"

    Returns:
        pandas.DataFrame with orientations
    """

    columns = "X Y Z G_x G_y G_z formation group".split()
    orientations = pd.DataFrame(columns=columns)

    for grp in interfaces[group].unique():
        f = interfaces[group] == grp

        v = interfaces[f]["X Y Z".split()].values
        C, N = plane_fit(v.T)
        #         v = (v.T - v.mean(axis=1)) / (v.max(axis=1) - v.min(axis=1))
        #         _, N = standard_fit(v.T)

        orientations = orientations.append(pd.Series([C[0], C[1], C[2],
                                                      N[0], N[1], N[2],
                                                      interfaces[interfaces[group] == grp]["formation"].unique()[0],
                                                      grp], columns),
                                           ignore_index=True)

    orientations["polarity"] = 1

    return orientations


def plane_fit(points):
    """
    Fit plane to points in PointSet
    Fit an d-dimensional plane to the points in a point set.
    adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    Args:
        point_list (array_like): array of points XYZ

    Returns:
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
    """

    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    centroid = points.mean(axis=1)
    x = points - centroid[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    normal = svd(M)[0][:, -1]
    # return ctr, svd(M)[0][:, -1]
    if normal[2] < 0:
        normal = -normal

    return centroid, normal