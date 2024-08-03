import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from mfsetup.fileio import check_source_files
from mfsetup.grid import get_ij


def read_observation_data(f=None, column_info=None,
                          column_mappings=None):

    df = pd.read_csv(f)
    df.columns = [s.lower() for s in df.columns]
    df['file'] = f
    xcol = column_info.get('x_location_col', 'x')
    ycol = column_info.get('y_location_col', 'y')
    obstype_col = column_info.get('obstype_col', 'obs_type')
    rename = {xcol: 'x',
              ycol: 'y',
              }
    if obstype_col is not None:
        rename.update({obstype_col.lower(): 'obs_type'})
        print('    observation type col: {}'.format(obstype_col))
    else:
        print('    no observation type col specified; observations assumed to be heads')
    if column_mappings is not None:
        for k, v in column_mappings.items():
            if not isinstance(v, list):
                v = [v]
            for vi in v:
                rename.update({vi.lower(): k.lower()})
                if vi in df.columns:
                    print('    observation label column: {}'.format(vi))
    if xcol is None or xcol.lower() not in rename:  # df.columns:
        raise ValueError("Column {} not in {}; need to specify x_location_col in config file"
                         .format(xcol, f))
    else:
        print('    x location col: {}'.format(xcol))
    if ycol is None or ycol.lower() not in rename:  # df.columns:
        raise ValueError("Column {} not in {}; need to specify y_location_col in config file"
                         .format(ycol, f))
    else:
        print('    y location col: {}'.format(ycol))
    df.rename(columns=rename, inplace=True)
    # force observation names to strings
    obsname_cols = ['obsname', 'hydlbl']
    for c in obsname_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def setup_head_observations(model, filenames=None,
                            obs_package='hyd', column_mappings=None,
                            obsname_column='obsname',
                            x_location_col='x',
                            y_location_col='y',
                            obsname_character_limit=40,
                            drop_observations=None, iobs_domain=None,
                            **kwargs):

    # set a 14 character obsname limit for the hydmod package
    # https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/index.html?hyd.htm
    # 40 character limit for MODFLOW-6 (see IO doc)


    # TODO: read head observation data using TabularSourceData instead
    if filenames is None:
        filenames = kwargs.get('filename')
        if filenames is None:
            print("No data for the Observation (OBS) utility.")
            return
    obs_info_files = filenames

    # get obs_info_files into dictionary format
    # filename: dict of column names mappings
    default_inputs = {
            'x_location_col': x_location_col,
            'y_location_col': y_location_col,
        }

    if isinstance(obs_info_files, str):
        obs_info_files = [obs_info_files]
    if isinstance(obs_info_files, list):
        obs_info_files = {f: default_inputs
                          for f in obs_info_files}
    elif isinstance(obs_info_files, dict):
        for k, v in obs_info_files.items():
            if v is None:
                obs_info_files[k] = default_inputs

    check_source_files(obs_info_files.keys())
    # dictionaries mapping from obstypes to hydmod input
    pckg = {'LK': 'BAS',  # head package for high-K lakes; lake package lakes get dropped
            'GW': 'BAS',
            'head': 'BAS',
            'lake': 'BAS',
            'ST': 'SFR',
            'flux': 'SFR'
            }
    arr = {'LK': 'HD',  # head package for high-K lakes; lake package lakes get dropped
           'GW': 'HD',
           'ST': 'SO',
           'flux': 'SO'
           }
    print('Reading observation files...')
    dfs = []
    for f, column_info in obs_info_files.items():
        print(f)
        df = read_observation_data(f, column_info,
                                   column_mappings=column_mappings)
        if 'obs_type' in df.columns and 'pckg' not in df.columns:
            df['pckg'] = [pckg.get(s, 'BAS') for s in df['obs_type']]
        elif 'pckg' not in df.columns:
            df['pckg'] = 'BAS'  # default to getting heads
        if 'obs_type' in df.columns and 'intyp' not in df.columns:
            df['arr'] = [arr.get(s, 'HD') for s in df['obs_type']]
        elif 'arr' not in df.columns:
            df['arr'] = 'HD'
        df['intyp'] = ['I' if p == 'BAS' else 'C' for p in df['pckg']]
        df[obsname_column] = df[obsname_column].astype(str).str.lower()

        dfs.append(df[['pckg', 'arr', 'intyp', 'x', 'y', obsname_column, 'file']])
    df = pd.concat(dfs, axis=0)

    print('\nCulling observations to model area...')
    df['geometry'] = [Point(x, y) for x, y in zip(df.x, df.y)]
    l, r, t, b = model.modelgrid.extent
    bbox = box(l, b, r, t)
    within = [g.within(bbox) for g in df.geometry]
    df = df.loc[within].copy()

    # make unique observation names for each model layer; applying the character limit
    # preserve end of obsname, truncating initial characters as needed
    # (for observations based on lat-lon coordinates such as usgs, or other naming schemes
    #  where names share leading characters)
    prefix_character_limit = obsname_character_limit  # - 2
    df[obsname_column] = [obsname[-prefix_character_limit:] for obsname in df[obsname_column]]
    duplicated = df[obsname_column].duplicated(keep=False)
    # check for duplicate names after truncation
    if duplicated.sum() > 0:
        print('Warning- {} duplicate observation names encountered. First instance of each name will be used.'.format(
            duplicated.sum()))
        print(df.loc[duplicated, [obsname_column, 'file']])

    # make sure every head observation is in each layer
    non_heads = df.loc[df.arr != 'HD'].copy()
    heads = df.loc[df.arr == 'HD'].copy()
    heads0 = heads.groupby(obsname_column).first().reset_index()
    heads0[obsname_column] = heads0[obsname_column].astype(str)
    heads_all_layers = pd.concat([heads0] * model.modelgrid.nlay).sort_values(by=obsname_column)
    heads_all_layers['klay'] = list(range(model.modelgrid.nlay)) * len(heads0)
    heads_all_layers[obsname_column] = ['{}'.format(obsname)  # _{:.0f}'.format(obsname, k)
                                        for obsname, k in zip(heads_all_layers[obsname_column],
                                                              heads_all_layers['klay'])]
    df = pd.concat([heads_all_layers, non_heads], axis=0)

    # dtypes
    assert df[obsname_column].dtype == object
    df['klay'] = df.klay.astype(int)

    print('Culling observations to cells allowed by iobs_domain...')
    i, j = get_ij(model.modelgrid, df.x.values, df.y.values)
    df['i'], df['j'] = i, j

    keep = np.array([True] * len(df))
    if iobs_domain is not None:
        keep = np.ravel(iobs_domain[df['klay'].values, i, j] > 0)
    if np.any(~keep):
        print(f'dropped {np.sum(~keep)} of {len(df)} observations in cells with bcs.')
        df = df.loc[keep].copy()

    if drop_observations is not None:
        print('Dropping head observations specified in drop_observations...')
        df = df.loc[~df[obsname_column].astype(str).isin(drop_observations)]

    if obs_package == 'hyd':
        # get model locations
        xl, yl = model.modelgrid.get_local_coords(df.x.values, df.y.values)
        df['xl'] = xl
        df['yl'] = yl
        # drop observations located in inactive cels
        ibdn = model.bas6.ibound.array[df.klay.values, df.i.values, df.j.values]
        active = ibdn >= 1
        df.drop(['i', 'j'], axis=1, inplace=True)
    elif obs_package == 'obs':  # mf6 observation utility
        obstype = {'BAS': 'HEAD'}
        renames = {'pckg': 'obstype'}
        df.pckg.replace(obstype, inplace=True)
        df.rename(columns=renames, inplace=True)
        df['id'] = list(zip(df.klay, df.i, df.j))
        # drop observations located in inactive cels
        idm = model.idomain[df.klay.values, df.i.values, df.j.values]
        active = idm >= 1
        df.drop(['arr', 'intyp', 'i', 'j'], axis=1, inplace=True)
    df = df.loc[active].copy()
    return df


def make_obsname(name, unique_names={},
                 maxlen=13  # allows for number_yyyymm
                 ):
    """Make an observation name of maxlen characters or
    less, that is not in unique_names."""
    for i in range(len(name) - maxlen + 1):
        end = -i if i > 0 else None
        slc = slice(-(maxlen+i), end)
        if name[slc] not in unique_names:
            return name[slc]
    return name[-maxlen:]


def remove_inactive_obs(obs_package_instance):
    """Remove boundary conditions from cells that are inactive.

    Parameters
    ----------
    obs_package_instance : flopy Modflow-6 Observation package instance
    """
    model = obs_package_instance.parent
    idomain = model.dis.idomain.array
    if model.version != 'mf6':
        raise NotImplementedError(
            "obs.py::remove_inactive_obs(): "
            "Support for removing MODFLOW 2005 not implemented.")
    for obsfile, recarray in obs_package_instance.continuous.data.items():
        try:
            k, i, j = zip(*recarray['id'])
        except:
            # for now, only support obs defined by k, i, j cellids
            return
        # cull any observations in inactive cells
        is_active = idomain[k, i, j] > 0
        # update the flopy dataset
        obs_package_instance.continuous.set_data({obsfile: recarray[is_active]})
