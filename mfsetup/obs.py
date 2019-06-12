import pandas as pd


def read_observation_data(f, column_info, column_mappings=None):
    df = pd.read_csv(f)
    df.columns = [s.lower() for s in df.columns]
    df['file'] = f
    xcol = column_info.get('x_location_col')
    ycol = column_info.get('y_location_col')
    obstype_col = column_info.get('obstype_col')
    if xcol is None or xcol.lower() not in df.columns:
        raise ValueError("Column {} not in {}; need to specify x_location_col in config file"
                         .format(xcol, f))
    else:
        print('    x location col: {}'.format(xcol))
    if ycol is None or ycol.lower() not in df.columns:
        raise ValueError("Column {} not in {}; need to specify y_location_col in config file"
                         .format(ycol, f))
    else:
        print('    y location col: {}'.format(ycol))
    rename = {xcol.lower(): 'x',
              ycol.lower(): 'y'
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
    df.rename(columns=rename, inplace=True)
    return df