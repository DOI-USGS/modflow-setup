"""
Get connections between packages to keep 'er movin'
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sfrmaker.routing import find_path
from shapely.geometry import Point


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


def get_sfr_package_connections(gwfgwf_exchangedata,
                                reach_data1, reach_data2, distance_threshold=1000):
    """Connect SFR reaches between SFR packages in a pair of groundwater flow models linked
    by the GWFGWF (simulation-level mover) Package.

    Parameters
    ----------
    gwfgwf_exchangedata : flopy recarray or pandas DataFrame
        Exchange data from the GWFGWF package
        (listing cell connections between two groundwater flow models).
    reach_data1 : DataFrame, similar to sfrmaker.SFRData.reach_data
        SFR reach information for model 1 in gwfgwf_exchangedata.
        Must contain reach numbers and 'geometry' column of shapely geometries
        for each reach (can be LineStrings or Polygons)
    reach_data2 : DataFrame, similar to sfrmaker.SFRData.reach_data
        SFR reach information for model 2 in gwfgwf_exchangedata.
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
    gwfgwf_exchangedata = pd.DataFrame(gwfgwf_exchangedata)
    all_reach_data1 = reach_data1.copy()
    all_reach_data2 = reach_data2.copy()

    # add cellids to connect reaches with GWFGWF exchanges
    # ignore layers in case there are any mismatches
    # due to pinchouts at the different model resolutions
    # or different layer assignments by SFRmaker
    # and because we only expect one SFR reach at each i, j location
    gwfgwf_exchangedata['cellidm1_2d'] =\
        [(i, j) for k, i, j in gwfgwf_exchangedata['cellidm1']]
    gwfgwf_exchangedata['cellidm2_2d'] =\
        [(i, j) for k, i, j in gwfgwf_exchangedata['cellidm2']]
    for df in all_reach_data1, all_reach_data2:
        if 'cellid' not in df.columns and\
            {'k', 'i', 'j'}.intersection(df.columns):
                df['cellid_2d'] =\
                    list(df[['i', 'j']].itertuples(index=False, name=None))
        else:
            df['cellid_2d'] = [(i, j) for k, i, j in df['cellid']]
    reach_data1 = all_reach_data1.loc[all_reach_data1['cellid_2d'].isin(
        gwfgwf_exchangedata['cellidm1_2d'])]
    reach_data2 = all_reach_data2.loc[all_reach_data2['cellid_2d'].isin(
        gwfgwf_exchangedata['cellidm2_2d'])]

    # starting locations of each reach along the parent/inset interface
    reach_data1['reach_start'] = [Point(g.coords[0]) for g in reach_data1['geometry']]
    reach_data2['reach_start'] = [Point(g.coords[0]) for g in reach_data2['geometry']]

    # neighboring cells (in a structured grid)
    def neighbors(i, j):
        return {(i-1, j-1), (i-1, j), (i-1, j+1),
                (i, j+1), (i+1, j+1), (i+1, j),
                (i+1, j-1), (i, j-1)}

    # make the connections
    parent_to_inset = dict()
    inset_to_parent = dict()
    # record connection distances
    # for breaking circular routing cases
    parent_to_inset_distances = dict()
    inset_to_parent_distances = dict()
    for _, r in reach_data1.iterrows():
        # check for connections to the other model
        # if the next reach is in another cell
        # along the parent/inset model interface,
        # or if the curret reach is an outlet
        if r['outreach'] in reach_data1['rno'].values or r['outreach'] == 0:
            # if the outreach is in a neighboring cell,
            # assume it is correct
            # (that the next reach is not in the other model)
            out_reach = reach_data1.loc[reach_data1['rno'] == r['outreach']]
            if r['outreach'] != 0 and\
                out_reach['cellid_2d'].iloc[0] in neighbors(*r['cellid_2d']):
                continue
            # if the distance to the next reach is greater than 1 cell
            # consider this each to be an outlet
            reach_end = Point(r['geometry'].coords[-1])
            distances = reach_end.distance(reach_data2['reach_start'])
            min_distance = np.min(distances)
            if min_distance < distance_threshold:
                next_reach = reach_data2.iloc[np.argmin(distances)]
                parent_to_inset[r['rno']] = next_reach['rno']
                parent_to_inset_distances[r['rno']] = min_distance
        # next reach is somewhere else in the parent model
        else:
            continue
    # repeat for inset to parent connections
    for i, r in reach_data2.iterrows():
        # check for connections to the other model
        # if the next reach is in another cell
        # along the parent/inset model interface,
        # or if the curret reach is an outlet
        if r['outreach'] in reach_data2['rno'].values or r['outreach'] == 0:
            # if the outreach is in a neighboring cell,
            # assume it is correct
            # (that the next reach is not in the other model)
            out_reach = reach_data2.loc[reach_data2['rno'] == r['outreach']]
            if r['outreach'] != 0 and\
                out_reach['cellid_2d'].iloc[0] in neighbors(*r['cellid_2d']):
                continue
            # if the distance to the next reach is greater than 1 cell
            # consider this each to be an outlet
            reach_end = Point(r['geometry'].coords[-1])
            distances = reach_end.distance(reach_data1['reach_start'])
            min_distance = np.min(distances)
            if min_distance < distance_threshold:
                next_reach = reach_data1.iloc[np.argmin(distances)]
                inset_to_parent[r['rno']] = next_reach['rno']
                inset_to_parent_distances[r['rno']] = min_distance
        # next reach is somewhere else in this model
        else:
            continue
    # check for circular connections (going both ways between two models)
    # retain connection with the smallest distance
    delete_parent_to_inset_items = set()
    for parent_reach, inset_reach in parent_to_inset.items():
        if parent_reach in inset_to_parent.values():
            parent_to_inset_distance = parent_to_inset_distances[parent_reach]
            if inset_to_parent_distances.get(inset_reach) is not None:
                inset_to_parent_distance = inset_to_parent_distances[inset_reach]
                if inset_to_parent_distance < parent_to_inset_distance:
                    delete_parent_to_inset_items.add(parent_reach)
                elif parent_to_inset_distance < inset_to_parent_distance:
                    del inset_to_parent[inset_reach]
                else:
                    raise ValueError(
                        "Circular connection between SFR Packages in the Mover Package input.\n"
                        f"Connection distance between the end of parent reach {parent_reach} "
                        f"in parent model and start of inset reach {inset_reach} in inset model "
                        f"is equal to\nthe distance between the end of inset reach {inset_reach} "
                        f"and start of parent reach {parent_reach}.\nCheck input linework."
                        )
    for parent_reach in delete_parent_to_inset_items:
        del parent_to_inset[parent_reach]
    return parent_to_inset, inset_to_parent


def get_mover_sfr_package_input(parent, inset, gwfgwf_exchangedata):
    """Set up the MODFLOW-6 water mover package at the simulation level.
    Automate set-up of the mover between SFR packages in LGR parent and inset models.
    todo: automate set-up of mover between SFR and lakes (within a model).

    Parameters
    ----------
    gwfgwf_exchangedata : flopy recarray or pandas DataFrame
        Exchange data from the GWFGWF package
        (listing cell connections between two groundwater flow models).
        """

    grid_spacing = parent.dis.delc.array[0]
    connections = []
    # use 2x grid spacing for distance threshold
    # because reaches could be small fragments in opposite corners of two adjacent cells
    to_inset, to_parent = get_sfr_package_connections(gwfgwf_exchangedata,
                                                      parent.sfrdata.reach_data,
                                                      inset.sfrdata.reach_data,
                                                      distance_threshold=2*grid_spacing)
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
