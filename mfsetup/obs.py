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
    layer_col = column_info.get('layer_col', 'layer')
    rename = {xcol: 'x',
              ycol: 'y',
              }
    if obstype_col is not None:
        rename.update({obstype_col.lower(): 'obs_type'})
        print('    observation type col: {}'.format(obstype_col))
    else:
        print('    no observation type col specified; observations assumed to be heads')
    if layer_col is not None and layer_col.lower() in df.columns:
        rename.update({layer_col.lower(): 'layer'})
        print('    observation layer col: {}'.format(layer_col))
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


def parse_observation_layers(value, nlay=None):
    """Parse a user-specified layer value for a head observation into a
    list of zero-based layer indices.

    Parameters
    ----------
    value : scalar, list, or str
        Layer specification for a single head observation. May be a single
        layer (e.g. ``2``), a list of layers (e.g. ``[0, 2]``), or a string
        of one or more layers separated by commas or spaces (e.g. ``"0, 2"``),
        as might be read from a CSV column.
    nlay : int, optional
        Number of model layers. If provided, parsed layers are validated to be
        within ``range(nlay)``.

    Returns
    -------
    layers : list of int or None
        List of layer indices, or ``None`` if no layer was specified
        (missing/empty value), indicating that the observation should be
        placed in all layers.
    """
    # no layer specified -> fall back to all layers
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ('nan', 'none'):
            return None
        # allow comma- or whitespace-separated layers in a single string
        parts = [p for p in value.replace(',', ' ').split()]
        layers = [int(float(p)) for p in parts]
    elif isinstance(value, (list, tuple, np.ndarray)):
        layers = [int(v) for v in value]
    else:
        # single numeric value
        layers = [int(value)]
    if nlay is not None:
        invalid = [k for k in layers if k < 0 or k >= nlay]
        if any(invalid):
            raise ValueError(
                "Specified observation layer(s) {} are outside of the model "
                "layer range (0-{}).".format(invalid, nlay - 1))
    return layers


def setup_head_observations(model, filenames=None,
                            obs_package='hyd', column_mappings=None,
                            obsname_column='obsname',
                            x_location_col='x',
                            y_location_col='y',
                            layer_col='layer',
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
            'layer_col': layer_col,
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

        columns = ['pckg', 'arr', 'intyp', 'x', 'y', obsname_column, 'file']
        # carry through a user-specified layer column (heads only)
        if 'layer' in df.columns:
            columns.append('layer')
        dfs.append(df[columns])
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

    # expand each head observation across model layers
    # by default, every head observation is placed in all layers;
    # if a layer column was supplied (heads only), place the observation
    # only in the specified layer(s)
    nlay = model.modelgrid.nlay
    non_heads = df.loc[df.arr != 'HD'].copy()
    heads = df.loc[df.arr == 'HD'].copy()
    heads0 = heads.groupby(obsname_column).first().reset_index()
    heads0[obsname_column] = heads0[obsname_column].astype(str)
    # determine the layer(s) for each head observation
    specified_layers = None
    if 'layer' in heads0.columns:
        specified_layers = [parse_observation_layers(value, nlay)
                            for value in heads0['layer']]
    head_dfs = []
    for n, (_, row) in enumerate(heads0.iterrows()):
        if specified_layers is not None and specified_layers[n] is not None:
            klayers = specified_layers[n]
        else:
            klayers = list(range(nlay))
        obs_layers = pd.concat([row.to_frame().T] * len(klayers), ignore_index=True)
        obs_layers['klay'] = klayers
        head_dfs.append(obs_layers)
    heads_all_layers = pd.concat(head_dfs, ignore_index=True)
    heads_all_layers = heads_all_layers.sort_values(by=obsname_column)
    heads_all_layers[obsname_column] = heads_all_layers[obsname_column].astype(str)
    # drop the source layer column now that klay has been assigned
    if 'layer' in heads_all_layers.columns:
        heads_all_layers.drop('layer', axis=1, inplace=True)
    if 'layer' in non_heads.columns:
        non_heads.drop('layer', axis=1, inplace=True)
    df = pd.concat([heads_all_layers, non_heads], axis=0)

    # dtypes
    assert pd.api.types.is_string_dtype(df[obsname_column].dtype)
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
        df['pckg'] = df['pckg'].replace(obstype)
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
