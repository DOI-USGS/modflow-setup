import re
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import flopy
fm = flopy.modflow
from flopy.utils.mflistfile import ListBudget
from gisutils import shp2df, project, get_proj_str
from mfsetup.evaporation import hamon_evaporation
from mfsetup.grid import rasterize
from mfsetup.sourcedata import SourceData, aggregate_dataframe_to_stress_period
from mfsetup.units import convert_length_units, convert_temperature_units


def make_lakarr2d(grid, lakesdata,
                  include_ids, id_column='hydroid'):
    """
    Make a nrow x ncol array with lake package extent for each lake,
    using the numbers in the 'id' column in the lakes shapefile.
    """
    if isinstance(lakesdata, str):
        lakes = shp2df(lakesdata)
    elif isinstance(lakesdata, pd.DataFrame):
        lakes = lakesdata.copy()
    else:
        raise ValueError('unrecognized input for "lakesdata": {}'.format(lakesdata))
    id_column = id_column.lower()
    lakes.columns = [c.lower() for c in lakes.columns]
    lakes.index = lakes[id_column]
    lakes = lakes.loc[include_ids]
    lakes['lakid'] = np.arange(1, len(lakes) + 1)
    lakes['geometry'] = [Polygon(g.exterior) for g in lakes.geometry]
    arr = rasterize(lakes, grid=grid, id_column='lakid')

    # ensure that order of hydroids is unchanged
    # (used to match features to lake IDs in lake package)
    assert lakes[id_column].tolist() == include_ids
    return arr


def make_bdlknc_zones(grid, lakesshp, include_ids,
                      feat_id_column='feat_id',
                      lake_package_id_column='lak_id'):
    """
    Make zones for populating with lakebed leakance values. Same as
    lakarr, but with a buffer around each lake so that horizontal
    connections have non-zero values of bdlknc, and near-shore
    areas can be assigend higher leakance values.
    """
    print('setting up lakebed leakance zones...')
    t0 = time.time()
    if isinstance(lakesshp, str):
        lakes = shp2df(lakesshp)
    elif isinstance(lakesshp, pd.DataFrame):
        lakes = lakesshp.copy()
    else:
        raise ValueError('unrecognized input for "lakesshp": {}'.format(lakesshp))
    # Exterior buffer
    id_column = feat_id_column.lower()
    lakes.columns = [c.lower() for c in lakes.columns]
    exterior_buffer = 30  # m
    lakes.index = lakes[id_column]
    lakes = lakes.loc[include_ids]
    if lake_package_id_column not in lakes.columns:
        lakes[lake_package_id_column] = np.arange(1, len(lakes) + 1)
    # speed up buffer construction by getting exteriors once
    # and probably more importantly,
    # simplifying possibly complex geometries of lakes generated from 2ft lidar
    unbuffered_exteriors = [Polygon(g.exterior).simplify(5) for g in lakes.geometry]
    lakes['geometry'] = [g.buffer(exterior_buffer) for g in unbuffered_exteriors]
    arr = rasterize(lakes, grid=grid, id_column=lake_package_id_column)

    # Interior buffer for lower leakance, assumed to be 20 m around the lake
    interior_buffer = -20  # m
    lakes['geometry'] = [g.buffer(interior_buffer) for g in unbuffered_exteriors]
    arr2 = rasterize(lakes, grid=grid, id_column=lake_package_id_column)
    arr2 = arr2 * 100  # Create new ids for the interior, as multiples of 10

    arr[arr2 > 0] = arr2[arr2 > 0]
    # ensure that order of hydroids is unchanged
    # (used to match features to lake IDs in lake package)
    assert lakes[id_column].tolist() == list(include_ids)
    print('finished in {:.2f}s'.format(time.time() - t0))
    return arr


def make_bdlknc2d(lakzones, littoral_leakance, profundal_leakance):
    """Make a lakebed leakance array using piecewise-constant zones
    and universal values for littoral and profundal leakance."""
    bdlknc = np.zeros(lakzones.shape, dtype=float)
    bdlknc[(lakzones > 0) & (lakzones < 100)] = littoral_leakance
    bdlknc[lakzones >= 100] = profundal_leakance
    return bdlknc


def get_stage_observations(listfile, lake_numbers):
    mfllak = LakListBudget(listfile)
    df_flux, df_vol = mfllak.get_dataframes()
    obs = []
    for k, v in lake_numbers.items():
        df = df_flux.loc[df_flux.lake==k].copy()
        df['obsname'] = ['p{:.0f}_ts{:.0f}_{}'.format(r.per, r.timestep, v) for i, r in df.iterrows()]
        df['obsval'] = 100*(df['stage'].values - df['stage'].values[0])
        obs.append(df[['obsname', 'obsval']])
    obs = pd.concat(obs, axis=0)
    return obs


def get_flux_variable_from_config(variable, model):
    data = model.cfg['lak'][variable]
    # copy to all stress periods
    # single scalar value
    if np.isscalar(data):
        data = [data] * model.nper
    # flat list of global values by stress period
    elif isinstance(data, list) and np.isscalar(data[0]):
        if len(data) < model.nper:
            for i in range(model.nper - len(data)):
                data.append(data[-1])
    # dict of lists, or nested dict of values by lake, stress_period
    else:
        raise NotImplementedError('Direct input of {} by lake.'.format(variable))
    assert isinstance(data, list)
    return data


def setup_lake_info(model):

    # lake package must have a source_data block
    # (e.g. for supplying shapefile delineating lake extents)
    source_data = model.cfg.get('lak', {}).get('source_data')
    if source_data is None:
        return
    lakesdata = model.load_features(**source_data['lakes_shapefile'])
    lakesdata_proj_str = get_proj_str(source_data['lakes_shapefile']['filename'])
    id_column = source_data['lakes_shapefile']['id_column'].lower()
    name_column = source_data['lakes_shapefile'].get('name_column', 'name').lower()
    nlakes = len(lakesdata)

    # save a lookup file mapping lake ids to hydroids
    centroids = project([g.centroid for g in lakesdata.geometry],
                        lakesdata_proj_str, 'epsg:4269')
    df = pd.DataFrame({'lak_id': np.arange(1, nlakes + 1),
                       'feat_id': lakesdata[id_column].values,
                       'name': lakesdata[name_column].values,
                       'latitude': [c.y for c in centroids],
                       'geometry': lakesdata['geometry']
                       })
    df.drop('geometry', axis=1).to_csv(model.cfg['lak']['output_files']['lookup_file'],
              index=False)
    return df


def setup_lake_fluxes(model):

    # setup empty dataframe
    variables = ['precipitation', 'evaporation', 'runoff', 'withdrawal']
    columns = ['per', 'lak_id'] + variables
    nlakes = len(model.lake_info['lak_id'])
    df = pd.DataFrame(np.zeros((nlakes * model.nper, len(columns)), dtype=float),
                      columns=columns)
    df['per'] = list(range(model.nper)) * nlakes
    df['lak_id'] = sorted(model.lake_info['lak_id'].tolist() * model.nper)

    # option 1; precip and evaporation specified directly
    # values are assumed to be in model units
    for variable in variables:
        if variable in model.cfg['lak']:
            values = get_flux_variable_from_config(variable, model)
            # repeat values for each lake
            df[variable] = values * len(model.lake_info['lak_id'])

    # option 2; precip and temp specified from PRISM output
    # compute evaporation from temp using Hamon method
    if 'climate' in model.cfg['lak']['source_data']:
        cfg = model.cfg['lak']['source_data']['climate']
        format = cfg.get('format', 'csv').lower()
        if format == 'prism':
            sd = PrismSourceData.from_config(cfg, dest_model=model)
            data = sd.get_data()
            tmid = data['start_datetime'] + (data['end_datetime'] - data['start_datetime'])/2
            data['day_of_year'] = tmid.dt.dayofyear
            id_lookup = {'feat_id': dict(zip(model.lake_info['lak_id'],
                                             model.lake_info['feat_id']))}
            id_lookup['lak_id'] = {v:k for k, v in id_lookup['feat_id'].items()}
            for col in ['lak_id', 'feat_id']:
                if len(set(data.lake_id).intersection(model.lake_info[col])) > 0:
                    latitude = dict(zip(model.lake_info[col],
                                        model.lake_info['latitude']))
                    other_col = 'lak_id' if col == 'feat_id' else 'feat_id'
                    data[other_col] = [id_lookup[other_col][id] for id in data['lake_id']]
            data['latitude'] = [latitude[id] for id in data['lake_id']]
            data['evaporation'] = hamon_evaporation(data['day_of_year'],
                                             data['temp'],  # temp in C
                                             data['latitude'],  # DD
                                             dest_length_units=model.length_units)
            # update flux dataframe
            for c in columns:
                if c in data:
                    df[c] = data[c]

        else:
            # TODO: option 3; general csv input for lake fluxes
            raise NotImplementedError('General csv input for lake fluxes')
    # compute a value to use for high-k lake recharge, for each stress period
    per_means = df.groupby('per').mean()
    highk_lake_rech = dict(per_means['precipitation'] - per_means['evaporation'])
    df['highk_lake_rech'] = [highk_lake_rech[per] for per in df.per]
    return df


class PrismSourceData(SourceData):
    """Subclass for handling tabular source data that
    represents a time series."""

    def __init__(self, filenames, period_stats='mean', id_column='lake_id',
                 dest_temperature_units='celsius',
                 dest_model=None):
        SourceData.__init__(self, filenames=filenames,
                            dest_model=dest_model)

        self.id_column = id_column
        self.period_stats = period_stats
        self.data_columns = ['precipitation', 'temp']
        self.datetime_column = 'datetime'
        self.dest_temperature_units = dest_temperature_units

    def parse_header(self, f):
        meta = {}
        with open(f) as src:
            for i, line in enumerate(src):
                if 'Location' in line:
                    _, _, lat, _, lon, _, _ = line.strip().split()
                if 'Date' in line:
                    names = line.strip().split(',')
                    meta['length_units'] = names[1].split()[1].strip('()')
                    meta['temperature_units'] = names[2].split()[-1].strip('()')
                    names = [self.datetime_column] + self.data_columns
                    break
        self.data_columns = names[1:]
        meta['latitude'] = float(lat)
        meta['longitude'] = float(lon)
        meta['column_names'] = names
        meta['skiprows'] = i + 1
        meta['temp_conversion'] = convert_temperature_units(meta['temperature_units'],
                                                            self.dest_temperature_units)
        return meta

    def get_data(self):

        # aggregate the data from multiple files
        dfs = []
        for id, f in self.filenames.items():
            meta = self.parse_header(f)
            df = pd.read_csv(f, skiprows=meta['skiprows'],
                             header=None, names=meta['column_names'])
            df.index = pd.to_datetime(df[self.datetime_column])
            df['start_datetime'] = df.index
            # check if data are monthly
            ndays0 = (df.index[1] - df.index[0]).days
            ismonthly = ndays0 >=28 & ndays0 <=31
            if ismonthly:
                ndays = df.index.days_in_month
                df['end_datetime'] = df['start_datetime'] + pd.to_timedelta(ndays, unit='D')
            elif ndays0 == 1:
                ndays = 1
                df['end_datetime'] = df['start_datetime']
            else:
                raise ValueError("Check {}; only monthly or daily values supported.")

            # convert precip to model units
            # assumes that precip is monthly values
            mult = convert_length_units(meta['length_units'],
                                        self.dest_model.length_units)
            df[meta['column_names'][1]] = df[meta['column_names'][1]] * mult/ndays

            # convert temperatures to C
            df[meta['column_names'][2]] = meta['temp_conversion'](df[meta['column_names'][2]])

            # record lake ID
            df[self.id_column] = id
            dfs.append(df)
        df = pd.concat(dfs)

        # sample values to model stress periods
        starttimes = self.dest_model.perioddata['start_datetime'].copy()
        endtimes = self.dest_model.perioddata['end_datetime'].copy()

        # if period ends are specified as the same as the next starttime
        # need to subtract a day, otherwise
        # pandas will include the first day of the next period in slices
        endtimes_equal_startimes = np.all(endtimes[:-1].values == starttimes[1:].values)
        #if endtimes_equal_startimes:
        #    endtimes -= pd.Timedelta(1, unit='d')

        period_data = []
        current_stat = None
        for kper, (start, end) in enumerate(zip(starttimes, endtimes)):
            # missing (period) keys default to 'mean';
            # 'none' to explicitly skip the stress period
            period_stat = self.period_stats.get(kper, current_stat)
            current_stat = period_stat
            aggregated = aggregate_dataframe_to_stress_period(df,
                                                              start_datetime=start,
                                                              end_datetime=end,
                                                              period_stat=period_stat,
                                                              id_column=self.id_column,
                                                              data_column=self.data_columns
                                                              )
            aggregated['per'] = kper
            period_data.append(aggregated)
        dfm = pd.concat(period_data)
        dfm.sort_values(by=['per', self.id_column], inplace=True)
        return dfm.reset_index(drop=True)


class LakListBudget(ListBudget):
    """

    """
    in_fieldnames = {'precip', 'total_runoff', 'runoff',
                     'gw_inflow', 'sw_inflow', 'connected_lake_influx'}
    out_fieldnames = {'evaporation', 'uzf_infil_from_lake', 'gw_outflow', 'sw_outflow', 'water_use'}

    def __init__(self, file_name, budgetkey=None, timeunit='days'):
        ListBudget.__init__(self, file_name, budgetkey=budgetkey, timeunit=timeunit)
        self.get_data()

    def set_budget_key(self):
        self.budgetkey = 'VOLUMETRIC BUDGET FOR ENTIRE MODEL'
        return

    def get_data(self):

        lakkey = 'HYDROLOGIC BUDGET SUMMARIES FOR SIMULATED LAKES'
        tskey = 'PERIOD', 'TIME STEP LENGTH'
        self.inc, self.cum = [], []
        with open(self.file_name) as self.f:
            for line in self.f:
                if tskey[0] in line and tskey[1] in line:
                    self._parse_kstp_kper(line)
                if lakkey in line:
                    self._parse_lak_budget()

    def get_dataframes(self, start_datetime='1-1-1970', diff=False):

        try:
            import pandas as pd
        except Exception as e:
            raise Exception(
                    "ListBudget.get_dataframe() error import pandas: " + \
                    str(e))

        if not self._isvalid:
            return None
        '''
        totim = self.get_times()
        if start_datetime is not None:
            totim = totim_to_datetime(totim,
                                      start=pd.to_datetime(start_datetime),
                                      timeunit=self.timeunit)
        '''
        df_flux = pd.DataFrame(self.inc).groupby(['per', 'timestep', 'lake']).max().reset_index()
        # figure out the column names
        in_components = list(self.in_fieldnames.intersection(set(df_flux.columns)))
        out_components = list(self.out_fieldnames.intersection(set(df_flux.columns)))
        df_flux['total_in'] = df_flux[in_components].sum(axis=1)
        df_flux['total_out'] = df_flux[out_components].sum(axis=1)
        df_vol = pd.DataFrame(self.cum).groupby('lake').max()
        return df_flux, df_vol

    def get_stage_dataframe(self, start_datetime='1-1-1970', diff=False):
        try:
            import pandas as pd
        except Exception as e:
            raise Exception(
                    "ListBudget.get_dataframe() error import pandas: " + \
                    str(e))

        if not self._isvalid:
            return None
        '''
        totim = self.get_times()
        if start_datetime is not None:
            totim = totim_to_datetime(totim,
                                      start=pd.to_datetime(start_datetime),
                                      timeunit=self.timeunit)
        '''
        components = ['stage', 'volume', 'timestep_surface_area']
        df = pd.DataFrame(self.inc).groupby('lake').max()
        return df[components]

    def _parse_kstp_kper(self, line):
        """Get the zero-based time step and stress period."""
        line = line.lower()
        self._c_kstp = int(line.split("time step")[1].strip(' ,')) - 1
        self._c_kper = int(line.split('period')[1].strip().split()[0]) - 1
        # and total time
        line = self.f.readline().lower()
        self._c_totim = float(line.split('total simulation time')[1].strip())

    def _parse_lak_budget(self):

        nextline = self.f.readline()
        if 'present time step' in nextline.lower():
            results = []
            for i in range(3):
                results += self._parse_block()
            self.inc += results
        elif 'since initial time' in nextline.lower():
            results = []
            for i in range(3):
                results += self._parse_block()
            self.cum += results

    def _parse_block(self):
        """Parse a block of Lake Budget output."""

        def split_whitespace(line):
            return [l.replace(' ', '_').strip('_')
                    for l in re.split(r'\s{2,}', line.strip())]

        while True:
            line = self.f.readline().lower()
            if line.strip().strip('-') != '':
                break
        names = split_whitespace(line)
        line2 = self.f.readline().lower()
        if line2.split()[0] != 'lake':
            line = line2.strip()
        else:
            line1 = ['']
            for name in names:
                if 'ground' in name.lower() and 'water' in name.lower():
                    line1 += ['gw', 'gw']
                elif 'surface' in name.lower() and 'water' in name.lower():
                    line1 += ['sw', 'sw']
                elif 'setage' in name.lower() and 'change' in name.lower():
                    line1 += ['stage_change', 'stage_change']
                else:
                    line1.append(name.replace('.', '').lower())
            line2 = split_whitespace(line2)
            names = ['_'.join([l1, l2]).strip('_').lower().replace('.', '').replace('-', '')
                     for l1, l2 in zip(line1, line2)]
            line = self.f.readline().strip()
        names = ['per', 'timestep', 'totim'] + names
        records = []
        while True:
            if line.strip('-') == '':
                break
            else:
                line = split_whitespace(line)
                values = [self._c_kper, self._c_kstp, self._c_totim, int(line[0])]
                for s in line[1:]:
                    try:
                        values.append(float(s))
                    except ValueError:
                        values.append(np.nan)
                records.append(OrderedDict(zip(names, values)))
                line = self.f.readline().strip()
        return records