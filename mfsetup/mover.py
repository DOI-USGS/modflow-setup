"""
Get connections between packages to keep 'er movin'
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def get_connections(from_features, to_features, distance_threshold=250):
    """Given two sequences of shapely geometries, return a dictionary
    of the (index position of the) elements in from_features (keys)
    and elements in to_features (values) that are less than distance_threshold apart.


    Parameters
    ----------
    from_features : sequence of shapely geometries
    to_features : sequence of shapely geometries

    Returns
    -------
    connections : dict
        {index in from_features : index in to_features}

    """
    x1, y1 = zip(*[g.centroid.coords[0] for g in from_features])
    x2, y2 = zip(*[g.centroid.coords[0] for g in to_features])
    points1 = np.array([x1, y1]).transpose()
    points2 = np.array([x2, y2]).transpose()
    # compute a matrix of distances between all points in vectors points1 and points2
    distances = cdist(points1, points2)
    # for each point in points1, get the closest point in points2
    # by getting the row, column locs where the distances are within the threshold
    # and then making a dictionary of row keys (points1) and column values (points2)
    connections = dict(np.transpose(np.where(distances < distance_threshold)))
    return connections


def get_sfr_package_connections(reach_data1, reach_data2, distance_threshold=1000):
    """Connect SFR reaches between two packages (for example, in a parent and inset model).
    Connections are made when a headwater reach in one package is within distance_threshold
    of an outlet in the other package.

    Parameters
    ----------
    reach_data1 : DataFrame, similar to sfrmaker.SFRData.reach_data
        Reach information for first package to connect.
        Must contain reach numbers and 'geometry' column of shapely geometries
        for each reach (can be LineStrings or Polygons)
    reach_data2 : DataFrame, similar to sfrmaker.SFRData.reach_data
        Reach information for second package to connect.
        Must contain reach numbers and 'geometry' column of shapely geometries
        for each reach (can be LineStrings or Polygons)
    distance_threshold : float
        Distance, in units of shapely geometries in reach data tables (usually meters)
        within which to look for connections.

    Returns
    -------
    connections1 : dictionary of connections from package 1 to package 2
    connections2 : dictionary of connections from package 2 to package 1
    """
    outlets1 = reach_data1.loc[reach_data1.outreach == 0]
    headwaters1 = set(reach_data1.rno).difference(reach_data1.outreach)
    headwaters1 = reach_data1.loc[reach_data1.rno.isin(headwaters1)]
    outlets2 = reach_data2.loc[reach_data2.outreach == 0]
    headwaters2 = set(reach_data2.rno).difference(reach_data2.outreach)
    headwaters2 = reach_data2.loc[reach_data2.rno.isin(headwaters2)]

    # get the connections between outlets and headwaters that are less than distance_threshold
    connections1_idx = get_connections(outlets1.geometry, headwaters2.geometry,
                                       distance_threshold=distance_threshold)
    connections2_idx = get_connections(outlets2.geometry, headwaters1.geometry,
                                       distance_threshold=distance_threshold)
    # map those back to actual reach numbers
    connections1 = {outlets1.rno.values[k]: headwaters2.rno.values[v]
                    for k, v in connections1_idx.items()}
    connections2 = {outlets2.rno.values[k]: headwaters1.rno.values[v]
                    for k, v in connections2_idx.items()}

    return connections1, connections2


def get_mover_sfr_package_input(parent, inset, convert_to_zero_based=True):

    grid_spacing = parent.dis.delc.array[0]
    connections = []
    to_inset, to_parent = get_sfr_package_connections(parent.sfrdata.reach_data,
                                                      inset.sfrdata.reach_data,
                                                      distance_threshold=grid_spacing)
    # convert to zero-based if reach_data aren't
    # corrections are quantities to subtract off
    parent_rno_correction = parent.sfrdata.reach_data.rno.min()
    inset_rno_correction = inset.sfrdata.reach_data.rno.min()

    for parent_reach, inset_reach in to_inset.items():
        rec = {'mname1': parent.name,
               'pname1': parent.sfr.package_name,
               'id1': parent_reach - parent_rno_correction,
               'mname2': inset.name,
               'pname2': inset.sfr.package_name,
               'id2': inset_reach - inset_rno_correction,
               'mvrtype': 'factor',  # see MF-6 IO documentation
               'value': 1.0  # move 100% of the water from parent_reach to inset_reach
               }
        connections.append(rec.copy())

    for inset_reach, parent_reach in to_parent.items():
        rec = {'mname1': inset.name,
               'pname1': inset.sfr.package_name,
               'id1': inset_reach - inset_rno_correction,
               'mname2': parent.name,
               'pname2': parent.sfr.package_name,
               'id2': parent_reach - parent_rno_correction,
               'mvrtype': 'factor',  # see MF-6 IO documentation
               'value': 1.0  # move 100% of the water from parent_reach to inset_reach
               }
        connections.append(rec.copy())
    packagedata = pd.DataFrame(connections)
    return packagedata
