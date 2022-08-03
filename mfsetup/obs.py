import numpy as np
import pandas as pd
from shapely.geometry import Point

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


def setup_head_observations(model, obs_info_files=None,
                            format='hyd',
                            obsname_column='obsname'):

    self = model
    package = format
    source_data_config = self.cfg[package]['source_data']

    # set a 14 character obsname limit for the hydmod package
    # https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/index.html?hyd.htm
    # 40 character limit for MODFLOW-6 (see IO doc)
    obsname_character_limit = 40
    if format == 'hyd':
        obsname_character_limit = 14

    # TODO: read head observation data using TabularSourceData instead
    if obs_info_files is None:
        for key in 'filename', 'filenames':
            if key in source_data_config:
                obs_info_files = source_data_config[key]
        if obs_info_files is None:
            print("No data for the Observation (OBS) utility.")
            return

    # get obs_info_files into dictionary format
    # filename: dict of column names mappings
    if isinstance(obs_info_files, str):
        obs_info_files = [obs_info_files]
    if isinstance(obs_info_files, list):
        obs_info_files = {f: self.cfg[package]['default_columns']
                          for f in obs_info_files}
    elif isinstance(obs_info_files, dict):
        for k, v in obs_info_files.items():
            if v is None:
                obs_info_files[k] = self.cfg[package]['default_columns']

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
        column_mappings = self.cfg[package]['source_data'].get('column_mappings')
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
    within = [g.within(self.bbox) for g in df.geometry]
    df = df.loc[within].copy()

    print('Dropping head observations that coincide with boundary conditions...')
    i, j = get_ij(self.modelgrid, df.x.values, df.y.values)
    df['i'], df['j'] = i, j

    #islak = self.lakarr[0, i, j] != 0
    # for now, discard any head observations in same (i, j) column of cells
    # as a non-well boundary condition
    # lake package lakes
    has_lak = np.any(self.isbc[:, i, j] == 1, axis=0)
    # non lake, non well BCs
    # (high-K lakes are excluded, since we may want head obs at those locations,
    #  to serve as pseudo lake stage observations)
    has_bc = has_lak | np.any(self.isbc[:, i, j] > 2, axis=0)
    if any(has_bc):
        print(f'dropped {np.sum(has_bc)} of {len(df)} observations in cells with bcs.')
    df = df.loc[~has_bc].copy()

    drop_obs = self.cfg[package].get('drop_observations', [])
    if len(drop_obs) > 0:
        print('Dropping head observations specified in {}...'.format(self.cfg.get('filename', 'config file')))
        df = df.loc[~df[obsname_column].astype(str).isin(drop_obs)]

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
    heads_all_layers = pd.concat([heads0] * self.nlay).sort_values(by=obsname_column)
    heads_all_layers['klay'] = list(range(self.nlay)) * len(heads0)
    heads_all_layers[obsname_column] = ['{}'.format(obsname)  # _{:.0f}'.format(obsname, k)
                                        for obsname, k in zip(heads_all_layers[obsname_column],
                                                              heads_all_layers['klay'])]
    df = pd.concat([heads_all_layers, non_heads], axis=0)

    # dtypes
    assert df[obsname_column].dtype == np.object
    df['klay'] = df.klay.astype(int)

    if format == 'hyd':
        # get model locations
        xl, yl = self.modelgrid.get_local_coords(df.x.values, df.y.values)
        df['xl'] = xl
        df['yl'] = yl
        # drop observations located in inactive cels
        ibdn = model.bas6.ibound.array[df.klay.values, df.i.values, df.j.values]
        active = ibdn >= 1
        df.drop(['i', 'j'], axis=1, inplace=True)
    elif format == 'obs':  # mf6 observation utility
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
