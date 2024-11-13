import numbers
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from flopy.utils import binaryfile as bf
from gisutils import get_values_at_points, project, shp2df
from scipy.interpolate import griddata
from shapely.geometry import Point

import xarray as xr
from mfsetup.discretization import (
    fill_cells_vertically,
    fill_empty_layers,
    fix_model_layer_conflicts,
    get_layer,
    populate_values,
    verify_minimum_layer_thickness,
    weighted_average_between_layers,
)
from mfsetup.fileio import save_array, setup_external_filepaths
from mfsetup.grid import get_ij, rasterize
from mfsetup.interpolate import (
    get_source_dest_model_xys,
    interp_weights,
    interpolate,
    regrid,
    regrid3d,
)
from mfsetup.mf5to6 import get_variable_name, get_variable_package_name
from mfsetup.tdis import (
    aggregate_dataframe_to_stress_period,
    aggregate_xarray_to_stress_period,
)
from mfsetup.units import convert_length_units, convert_time_units, convert_volume_units
from mfsetup.utils import get_input_arguments

renames = {'mult': 'multiplier',
           'elevation_units': 'length_units',
           'from_parent': 'from_source_model_layers',
           'data_column': 'data_columns'
           }


class SourceData:
    """Class for handling source_data specified in config file.

    Parameters
    ----------
    filenames :
    length_units :
    time_units :
    area_units :
    volume_units :
    datatype :
    dest_model :
    """
    def __init__(self, filenames=None, values=None, variable=None, length_units='unknown',
                 time_units='unknown',
                 area_units=None, volume_units=None,
                 datatype=None,
                 dest_model=None):
        """

        """
        self.filenames = filenames
        self.values = values
        self.variable = variable
        self.length_units = length_units
        self.area_units = area_units
        self.volume_units = volume_units
        self.time_units = time_units
        self.datatype = datatype
        self.dest_model = dest_model
        self.set_filenames(filenames)

    @property
    def unit_conversion(self):
        try:
            if np.issubdtype(self.source_array.dtype, np.integer):
                return 1.0
        except:
            pass
        # non-comprehensive list of dimensionless variables
        if self.variable in {'ibound', 'idomain', 'ss', 'sy', 'irch', 'iconvert', 'lkarr'}:
            return 1.0
        return self.length_unit_conversion / self.time_unit_conversion

    @property
    def length_unit_conversion(self):
        # data are lengths
        mult = convert_length_units(self.length_units,
                                    getattr(self.dest_model, 'length_units', 'unknown'))

        # data are areas
        if self.area_units is not None:
            raise NotImplementedError('Conversion of area units.')

        # data are volumes
        elif self.volume_units is not None:
            mult = convert_volume_units(self.volume_units,
                                 getattr(self.dest_model, 'length_units', 'unknown'))
        return mult

    @property
    def time_unit_conversion(self):
        return convert_time_units(self.time_units,
                                  getattr(self.dest_model, 'time_units', 'unknown'))

    def set_filenames(self, filenames):

        def normpath(f):
            if self.dest_model is not None and isinstance(f, str):
                path = os.path.join(self.dest_model._config_path, f)
                normpath = os.path.normpath(path)
                return normpath
            return f

        if isinstance(filenames, str):
            self.filenames = {0: normpath(filenames)}
        elif isinstance(filenames, list):
            self.filenames = {i: normpath(f) for i, f in enumerate(filenames)}
        elif isinstance(filenames, dict):
            self.filenames = {i: normpath(f) for i, f in filenames.items()}
        else:
            self.filenames = None

    @classmethod
    def from_config(cls, data, **kwargs):
        """Create a SourceData instance from a source_data
        entry read from an MFsetup configuration file.

        Parameters
        ----------
        data : str, list or dict
            Parse entry from the configuration file.
        type : str
            'array' for array data or 'tabular' for tabular data


        """
        data_dict = {}
        key = 'filenames'
        if isinstance(data, dict):
            data = data.copy()
            # rename keys for constructors
            for k, v in renames.items():
                if k in data.keys():
                    data[v] = data.pop(k)
            data_dict = data.copy()
            if key[:-1] in data_dict.keys(): # plural vs singular
                data_dict[key] = {0: data_dict.pop(key[:-1])}
            elif key in data_dict.keys():
                if isinstance(data_dict[key], list):
                    data_dict[key] = {i: f for i, f in enumerate(data_dict[key])}
            elif 'from_source_model_layers' in data_dict.keys():
                pass
            else:
                data_dict = {key: data_dict}
        elif isinstance(data, str):
            data_dict[key] = {0: data}
        elif isinstance(data, list):
            data_dict[key] = {i: f for i, f in enumerate(data)}
        else:
            raise TypeError("unrecognized input: {}".format(data))

        data_dict = get_input_arguments(data_dict, cls)
        kwargs = get_input_arguments(kwargs, cls)
        return cls(**data_dict, **kwargs)


class TransientSourceDataMixin():
    """Class for shared functionality among the SourceData subclasses
    that deal with transient data.
    """
    def __init__(self, period_stats, dest_model):

        # property attributes
        self._period_stats_input = period_stats
        self._period_stats = None

        # attributes
        self.dest_model = dest_model
        self.perioddata = dest_model.perioddata.sort_values(by='per').reset_index(drop=True)

    @property
    def period_stats(self):
        if self._period_stats is None:
            self._period_stats = self.get_period_stats()
        return self._period_stats

    @property
    def stress_period_mapping(self):
        # if there is a parent/source model,
        # get the mapping between the parent model and
        # inset/destination model stress periods {inset_kper: parent_kper}
        if self.dest_model.parent is not None:
            # for now, just assume one-to-one correspondance
            # between source and dest model stress periods
            return self.dest_model.parent_stress_periods
        # otherwise, just return a dictionary of the same
        # key, value pairs for consistency
        # with logic of subclass get_data() methods
        else:
            return dict(zip(self.perioddata['per'], self.perioddata['per']))

    def get_period_stats(self):
        """Populate each stress period with period_stat information
        for temporal resampling (tdis.aggregate_dataframe_to_stress_period and
        tdis.aggregate_xarray_to_stress_period methods), implementing default
        behavior for periods with unspecified start and end times.
        """

        perioddata = self.perioddata
        period_stats = {}
        period_stat_input = None
        for i, r in perioddata.iterrows():

            # get start end end datetimes from period_stats if provided
            start_datetime = None
            end_datetime = None

            # if there is no input for a period, reuse input for the last one
            if r.per not in self._period_stats_input:
                period_stats[r.per] = period_stat_input
            else:
                period_stat_input = self._period_stats_input[r.per]

            # dict of info for this period
            period_data_output = {}

            # set entries parsed as 'None' or 'none' to NoneType
            # if None was input for a period, skip it
            if isinstance(period_stat_input, str) and period_stat_input.lower() == 'none':
                period_stat_input = None
            if period_stat_input is None:
                period_stats[r.per] = None
                continue

            # if no start and end date are given in period_stats
            elif isinstance(period_stat_input, str):

                period_data_output['period_stat'] = period_stat_input
                # if the period is longer than one day, or transient
                # use start_datetime and end_datetime from perioddata
                if r.perlen > 1 or not r.steady:
                    period_data_output['start_datetime'] = r.start_datetime
                    period_data_output['end_datetime'] = r.end_datetime
                    if end_datetime == start_datetime:
                        period_data_output['end_datetime'] -= pd.Timedelta(1, unit=self.dest_model.time_units)
                # otherwise, for steady-state periods of one day
                # default to start and end datetime default as None
                # which will result in default aggregation of whole data file
                # by tdis.aggregate_dataframe_to_stress_period

            # aggregation time period defined by single string
            # e.g. 'august' or '2014-01'
            elif len(period_stat_input) == 2:
                period_data_output['period_stat'] = period_stat_input

            # aggregation time period defined by start and end dates
            elif len(period_stat_input) == 3:
                period_data_output['period_stat'] = period_stat_input
                period_data_output['start_datetime'] = period_stat_input[1]
                period_data_output['end_datetime'] = period_stat_input[2]

            else:
                raise ValueError('period_stat input for period {} not understood: {}'.format(r.per, period_stat_input))

            period_stats[r.per] = period_data_output
        return period_stats


class ArraySourceData(SourceData):
    """Subclass for handling array-based source data.

    Parameters
    ----------
    variable : str
        MODFLOW variable name (e.g. 'hk')
    filenames : list of file paths
    length_units : str
        'meters' or 'feet', etc.
    time_units : : str
        e.g. 'days'
    area_units : str
        e.g. 'm2', 'ft2', etc.
    volume_units : str
        e.g. 'm3', 'ft3', etc.
    datatype : str
        Type of array, following terminology used in flopy:
            array2d : e.g. model top
            array3d : e.g. model layer bottoms
            transient2d : e.g. recharge
            transient3d : e.g. head results
        (see flopy.datbase.DataType)
    dest_model : model instance
        Destination model that source data will be mapped to.

    Methods
    -------
    get_data : returns source data mapped to model grid,
               in a dictionary of 2D arrays
               (of same shape as destination model grid)
    """
    def __init__(self, variable, filenames=None, values=None, length_units='unknown', time_units='unknown',
                 dest_model=None, source_modelgrid=None, source_array=None,
                 from_source_model_layers=None, source_array_includes_model_top=False, datatype=None,
                 id_column=None, include_ids=None, column_mappings=None,
                 resample_method='linear',
                 vmin=-1e30, vmax=1e30, dtype=float,
                 multiplier=1.):

        SourceData.__init__(self, filenames=filenames, values=values,
                            variable=variable,
                            length_units=length_units, time_units=time_units,
                            datatype=datatype,
                            dest_model=dest_model)

        self.source_modelgrid = source_modelgrid
        self._source_mask = None
        if from_source_model_layers == {}:
            from_source_model_layers = None
        self.from_source_model_layers = from_source_model_layers
        self.source_array = None
        if source_array is not None:
            if len(source_array.shape) == 2:
                source_array = source_array.reshape(1, *source_array.shape)
            self.source_array = source_array.copy()
        self.source_array_includes_model_top = source_array_includes_model_top
        self.dest_modelgrid = getattr(self.dest_model, 'modelgrid', None)
        self.datatype = datatype
        self.id_column = id_column
        self.include_ids = include_ids
        self.column_mappings = column_mappings
        self.resample_method = resample_method
        self._interp_weights = None
        self.vmin = vmin
        self.vmax = vmax
        self.dtype = dtype
        self.mult = multiplier
        self.data = {}
        assert True

    @property
    def dest_source_layer_mapping(self):
        nlay = self.dest_model.nlay
        if self.from_source_model_layers is None:
            if self.datatype == 'array2d':
                return {0: 0}
            # if no layer mapping is provided
            # can't interpolate botm elevations (as they are the z values)
            # assume 1:1 layer mapping
            elif self.variable == 'botm':
                return dict(zip(range(self.source_modelgrid.nlay), range(nlay)))
            return self.dest_model.parent_layers
        elif self.from_source_model_layers is not None:
            nspecified = len(self.from_source_model_layers)
            if self.datatype == 'array3d' and nspecified != nlay:
                raise Exception("Variable should have {} layers "
                                "but {} are specified: {}"
                                .format(nlay, nspecified, self.from_source_model_layers))
            return self.from_source_model_layers
        elif self.filenames is not None:
            nspecified = len(self.filenames)
            if self.datatype == 'array3d' and nspecified != nlay:
                raise Exception("Variable should have {} layers "
                                "but only {} are specified: {}"
                                .format(nlay, nspecified, self.filenames))

    @property
    def interp_weights(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to pfl_nwt."""
        if self._interp_weights is None:
            source_xy, dest_xy = get_source_dest_model_xys(self.source_modelgrid,
                                                           self.dest_model,
                                                           source_mask=self._source_grid_mask)
            self._interp_weights = interp_weights(source_xy, dest_xy)
        return self._interp_weights

    @property
    def _source_grid_mask(self):
        """Boolean array indicating window in parent model grid (subset of cells)
        that encompass the pfl_nwt model domain. Used to speed up interpolation
        of parent grid values onto pfl_nwt grid."""
        if self._source_mask is None:
            mask = np.zeros((self.source_modelgrid.nrow,
                             self.source_modelgrid.ncol), dtype=bool)
            if self.dest_model.parent is not None:
                if self.dest_model.parent_mask.shape == self.source_modelgrid.xcellcenters.shape:
                    mask = self.dest_model.parent_mask
                else:
                    x, y = np.squeeze(self.dest_model.bbox.exterior.coords.xy)
                    pi, pj = get_ij(self.source_modelgrid, x, y)
                    pad = 3
                    i0, i1 = pi.min() - pad, pi.max() + pad
                    j0, j1 = pj.min() - pad, pj.max() + pad
                    mask[i0:i1, j0:j1] = True
            self._source_mask = mask
        return self._source_mask

    def regrid_from_source_model(self, source_array,
                                 mask=None,
                                 method='linear'):
        """Interpolate values in source array onto
        the destination model grid, using SpatialReference instances
        attached to the source and destination models.

        Parameters
        ----------
        source_array : ndarray
            Values from source model to be interpolated to destination grid.
            1 or 2-D numpy array of same sizes as a
            layer of the source model.
        mask : ndarray (bool)
            1 or 2-D numpy array of same sizes as a
            layer of the source model. True values
            indicate cells to include in interpolation,
            False values indicate cells that will be
            dropped.
        method : str ('linear', 'nearest')
            Interpolation method.
        """
        if mask is not None:
            return regrid(source_array, self.source_modelgrid, self.dest_modelgrid,
                          mask1=mask,
                          method=method)
        if method == 'linear':
            parent_values = source_array.flatten()[self._source_grid_mask.flatten()]
            regridded = interpolate(parent_values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = regrid(source_array, self.source_modelgrid, self.dest_modelgrid,
                               method='nearest')
        regridded = np.reshape(regridded, (self.dest_modelgrid.nrow,
                                           self.dest_modelgrid.ncol))
        return regridded

    def _read_array_from_file(self, filename):
        f = filename
        if isinstance(f, numbers.Number):
            data = f
        elif isinstance(f, str):
            # sample "source_data" that may not be on same grid
            # TODO: add bilinear and zonal statistics methods
            if any([f.lower().endswith(i) for i in ['asc', 'tif', 'tiff', 'geotiff', 'gtiff', 'vrt']]):
                arr = get_values_at_points(f,
                                           self.dest_model.modelgrid.xcellcenters.ravel(),
                                           self.dest_model.modelgrid.ycellcenters.ravel(),
                                           points_crs=self.dest_model.modelgrid.crs,
                                           method=self.resample_method)
                arr = np.reshape(arr, (self.dest_modelgrid.nrow,
                                       self.dest_modelgrid.ncol))
            elif f.endswith('.shp'):
                arr = rasterize(f, self.dest_modelgrid, id_column=self.id_column)
            # TODO: add code to interpret hds and cbb files
            # interpolate from source model using source model grid
            # otherwise assume the grids are the same

            # read numpy array on same grid
            # (load_array checks the shape)
            elif self.source_modelgrid is None:
                arr = self.dest_model.load_array(f)
            else:
                raise Exception('variable {}: unrecognized file type for array data input: {}'
                                .format(self.variable, f))

            assert arr.shape == self.dest_modelgrid.shape[1:]
            data = (arr * self.mult * self.unit_conversion).astype(self.dtype)
        return data

    def get_data(self):
        data = {}
        # start with any specified values,
        # interpolated to any layers without values
        # layers with filenames specified will be overwritten with data from the files
        if self.values is not None:
            data = populate_values(self.values, array_shape=(self.dest_modelgrid.nrow,
                                                             self.dest_modelgrid.ncol))

        if self.filenames is not None:
            for i, f in self.filenames.items():
                data[i] = self._read_array_from_file(f)

            # interpolate any missing arrays from consecutive files based on weights
            for i, arr in data.items():
                if np.isscalar(arr):
                    source_k = arr
                    weight0 = source_k - np.floor(source_k)
                    # get the next layers above and below that have data
                    source_k0 = int(np.max([k for k, v in data.items()
                                            if isinstance(v, np.ndarray) and k < i]))
                    source_k1 = int(np.min([k for k, v in data.items()
                                            if isinstance(v, np.ndarray) and k > i]))
                    data[i] = weighted_average_between_layers(data[source_k0],
                                                              data[source_k1],
                                                              weight0=weight0)

            # repeat last layer if length of data is less than number of layers
            # we probably don't want to do this for layer bottoms,
            # but may want to for aquifer properties
            if (self.variable != 'botm') and (self.datatype == 'array3d')\
                and (i < (self.dest_model.nlay - 1)):
                for j in range(i, self.dest_model.nlay):
                    data[j] = data[i]

        # regrid source data from another model
        elif self.source_array is not None:

            # interpolate by layer, using the layer mapping specified by the user
            #if self.from_source_model_layers is not None or self.datatype == 'array2d':
            if self.dest_source_layer_mapping is not None:
                for dest_k, source_k in self.dest_source_layer_mapping.items():
                    if source_k >= self.source_array.shape[0]:
                        continue
                    # destination model layers copied from source model layers
                    # if source_array has an extra layer, assume layer 0 is the model top
                    # (only included for weighted average)
                    # could use a better approach
                    # check source_k is a whole number to 4 decimal places
                    # and if is a layer in source_array
                    if np.round(source_k, 4) in range(self.source_array.shape[0]):
                        #if (self.source_array.shape[0] - self.dest_model.nlay == 1) and\
                        #    ((source_k + 1) < self.source_array.shape[0]):
                        #    source_k +=1
                        if self.source_array_includes_model_top and\
                            ((source_k + 1) < self.source_array.shape[0]):
                            source_k += 1
                        source_k = int(np.round(source_k, 4))
                        arr = self.source_array[source_k]
                    # destination model layers that are a weighted average
                    # of consecutive source model layers
                    else:
                        weight0 = source_k - np.floor(source_k)
                        source_k0 = int(np.floor(source_k))
                        # first layer in the average can't be negative
                        source_k0 = 0 if source_k0 < 0 else source_k0
                        source_k1 = int(np.ceil(source_k))
                        arr = weighted_average_between_layers(self.source_array[source_k0],
                                                              self.source_array[source_k1],
                                                              weight0=weight0)
                    # interpolate from source model using source model grid
                    # otherwise assume the grids are the same
                    if self.source_modelgrid is not None:
                        # exclude invalid values in interpolation from parent model
                        mask = self._source_grid_mask & (arr > self.vmin) & (arr < self.vmax)

                        regridded = self.regrid_from_source_model(arr,
                                                                  mask=mask,
                                                                  method=self.resample_method)

                    assert regridded.shape == self.dest_modelgrid.shape[1:]
                    data[dest_k] = regridded * self.mult * self.unit_conversion
            # general 3D interpolation based on the location of parent and inset model cells
            else:
                # tile the mask to nlay x nrow x ncol
                in_window = np.tile(self._source_grid_mask, (self.source_modelgrid.nlay, 1, 1))
                valid = (self.source_array > self.vmin) & (self.source_array < self.vmax)
                mask = valid & in_window
                heads = regrid3d(self.source_array, self.source_modelgrid, self.dest_modelgrid,
                                 mask1=mask, method='linear')
                data = {k: heads2d for k, heads2d in enumerate(heads)}

        # no files or source array provided
        else:
            raise ValueError("No files or source model grid provided.")

        self.data = data
        return data


class TransientArraySourceData(ArraySourceData, TransientSourceDataMixin):
    def __init__(self, filenames, variable, period_stats=None,
                 length_units='unknown', time_units='days',
                 dest_model=None, source_modelgrid=None, source_array=None,
                 from_source_model_layers=None, datatype=None,
                 resample_method='nearest', vmin=-1e30, vmax=1e30,
                 multiplier=1.
                 ):

        ArraySourceData.__init__(self, variable=None, filenames=filenames,
                                 length_units=length_units, time_units=time_units,
                                 dest_model=dest_model, source_modelgrid=source_modelgrid,
                                 source_array=source_array,
                                 from_source_model_layers=from_source_model_layers,
                                 datatype=datatype,
                                 resample_method=resample_method, vmin=vmin, vmax=vmax,
                                 multiplier=multiplier)
        TransientSourceDataMixin.__init__(self, period_stats=period_stats, dest_model=dest_model)

        self.variable = variable
        self.resample_method = resample_method

    def get_data(self):

        # get data from list of files; one per stress period
        # (files are assumed to be sorted)
        if self.filenames is not None:
            source_data = []
            for i, f in self.filenames.items():
                source_data.append(self._read_array_from_file(f))
            source_data = np.array(source_data)
            regrid = False  # data already regridded by _read_array_from_file

        # regrid source data from another model
        elif self.source_array is not None:
            source_data = self.source_array * self.unit_conversion * self.mult
            regrid = True

        # cast data to an xarray DataArray for time-sliceing
        # TODO: implement temporal resampling from source model
        # would follow logic of netcdf files, but trickier because steady-state periods need to be handled
        #da = transient2d_to_xarray(data, time)

        results = {}
        for dest_kper, source_kper in self.stress_period_mapping.items():
            data = source_data[source_kper].copy()
            if regrid:
                # sample the data onto the model grid
                resampled = self.regrid_from_source_model(data, method=self.resample_method)
            else:
                resampled = data
            # reshape results to model grid
            period_mean2d = resampled.reshape(self.dest_model.nrow,
                                              self.dest_model.ncol)
            results[dest_kper] = period_mean2d
        self.data = results
        return results


class NetCDFSourceData(ArraySourceData, TransientSourceDataMixin):
    def __init__(self, filenames, variable, period_stats,
                 length_units='unknown', time_units='days', crs=None,
                 dest_model=None, source_modelgrid=None,
                 from_source_model_layers=None, datatype='transient2d',
                 resample_method='nearest', vmin=-1e30, vmax=1e30
                 ):

        ArraySourceData.__init__(self, variable=None,
                                 length_units=length_units, time_units=time_units,
                                 dest_model=dest_model, source_modelgrid=source_modelgrid,
                                 from_source_model_layers=from_source_model_layers,
                                 datatype=datatype,
                                 resample_method=resample_method, vmin=vmin, vmax=vmax)
        TransientSourceDataMixin.__init__(self, period_stats=period_stats, dest_model=dest_model)

        if isinstance(filenames, dict):
            filenames = list(filenames.values())
        if isinstance(filenames, list):
            if len(filenames) > 1:
                raise NotImplementedError("Multiple NetCDF files not supported.")
            self.filename = filenames[0]
        else:
            self.filename = filenames
        self.variable = variable
        self.resample_method = resample_method
        self.dest_model = dest_model
        self.time_col = 'time'

        self._crs = None
        self._specified_crs = crs

        # set xy value arrays for source and dest. grids
        with xr.open_dataset(self.filename) as ds:
            x1, y1 = np.meshgrid(ds.x.values, ds.y.values)
            x1 = x1.ravel()
            y1 = y1.ravel()
            # reproject the netcdf coords
            # to the model CRS
            if self.crs is not None:
                if self.crs != self.dest_model.modelgrid.crs:
                    x1, y1 = project((x1, y1),
                                     self.crs,
                                     self.dest_model.modelgrid.crs)
            self.source_grid_xy = np.array([x1, y1]).transpose()
        x2 = self.dest_model.modelgrid.xcellcenters.ravel()
        y2 = self.dest_model.modelgrid.ycellcenters.ravel()
        self.dest_grid_xy = np.array([x2, y2]).transpose()

    @property
    def interp_weights(self):
        """For a given parent, only calculate interpolation weights
        once to speed up re-gridding of arrays to pfl_nwt."""
        if self._interp_weights is None:
            self._interp_weights = interp_weights(self.source_grid_xy,
                                                  self.dest_grid_xy)
        return self._interp_weights

    @property
    def crs(self):
        """Try to make a valid pyproj.CRS instance from
        coordinate reference system information in the
        NetCDF file. Assumes that CF-like grid_mapping
        information is stored in a 'crs' variable.

        """
        specified_crs = None
        ncfile_crs = None
        if self._specified_crs is not None:
            specified_crs = pyproj.CRS(self._specified_crs)
            self._specified_crs = specified_crs
        if self._crs is None:
            with xr.open_dataset(self.filename) as ds:
                crs_da = getattr(ds, 'crs', None)
            if crs_da is not None:
                grid_mapping = crs_da.attrs
                ncfile_crs = self.get_crs_from_grid_mapping(grid_mapping)
            if specified_crs is None:
                if ncfile_crs is None:
                    print('Could not create valid pyproj.CRS object'
                         f'from grid mapping infromation in {self.filename} '
                         f'Input in {self.filename} will be assumed to be in the '
                          'model coordinate reference system.')
                else:
                    self._crs = ncfile_crs
            elif (ncfile_crs is None) or (specified_crs == ncfile_crs):
                self._crs = specified_crs
            else:
                raise ValueError(f'Specified CRS is different than CRS in NetCDF file; check for consistency.\n'
                                 '{specified_crs}\n!=\n{ncfile_crs}')

        return self._crs

    def regrid_from_source(self, source_array,
                           method='linear'):
        """Interpolate values in source array onto
        the destination model grid, using SpatialReference instances
        attached to the source and destination models.

        Parameters
        ----------
        source_array : ndarray
            Values from source model to be interpolated to destination grid.
            1 or 2-D numpy array of same sizes as a
            layer of the source model.
        method : str ('linear', 'nearest')
            Interpolation method.
        """
        values = source_array.flatten()
        if method == 'linear':
            regridded = interpolate(values,
                                    *self.interp_weights)
        elif method == 'nearest':
            regridded = griddata(self.source_grid_xy, values, self.dest_grid_xy, method=method)
        regridded = np.reshape(regridded, (self.dest_model.nrow,
                                           self.dest_model.ncol))
        return regridded

    def get_data(self):

        # create an xarray dataset instance
        ds = xr.open_dataset(self.filename)
        data = ds[self.variable]

        # sample values to model stress periods
        results = {}
        for kper, period_stat in self.period_stats.items():
            if period_stat is None:
                continue
            aggregated = aggregate_xarray_to_stress_period(data,
                                                           datetime_coords_name=self.time_col,
                                                           **period_stat)
            # sample the data onto the model grid
            resampled = self.regrid_from_source(aggregated,
                                                method=self.resample_method)
            # reshape results to model grid
            period_mean2d = resampled.reshape(self.dest_model.nrow,
                                              self.dest_model.ncol)
            results[kper] = period_mean2d * self.unit_conversion
        self.data = results
        return results

    @staticmethod
    def get_crs_from_grid_mapping(grid_mapping):
        crs = None
        try:
            crs = pyproj.CRS.from_cf(grid_mapping)
        except:
            pass
        if 'crs_wkt' in grid_mapping:
            try:
                crs = pyproj.CRS(grid_mapping['crs_wkt'])
            except:
                pass
        # Soil Water Balance Code output
        # usually has a "proj4_string" entry
        if 'proj4_string' in grid_mapping:
            try:
                crs = pyproj.CRS(grid_mapping['proj4_string'])
            except:
                pass
        # could add more crazy try/excepts here
        return crs

class MFBinaryArraySourceData(ArraySourceData):
    """Subclass for handling MODFLOW binary array data
    that may come from another model."""
    def __init__(self, variable, filename=None,
                 length_units='unknown', time_units='unknown',
                 dest_model=None, source_modelgrid=None,
                 from_source_model_layers=None, stress_period=0,
                 datatype='transient3d',
                 resample_method='nearest', vmin=-1e30, vmax=1e30
                 ):

        ArraySourceData.__init__(self, variable=variable,
                                 length_units=length_units, time_units=time_units,
                                 dest_model=dest_model, source_modelgrid=source_modelgrid,
                                 from_source_model_layers=from_source_model_layers,
                                 datatype=datatype,
                                 resample_method=resample_method, vmin=vmin, vmax=vmax)

        self.filename = filename
        self.stress_period = stress_period

    @property
    def dest_source_layer_mapping(self):
        nlay = self.dest_model.nlay
        # if mapping between source and dest model layers isn't specified
        # use property from dest model
        # this will be the DIS package layer mapping if specified
        # otherwise same layering is assumed for both models
        if self.from_source_model_layers is None:
            return self.dest_model.parent_layers
        elif self.from_source_model_layers is not None:
            nspecified = len(self.from_source_model_layers)
            if nspecified != nlay:
                raise Exception("Variable should have {} layers "
                                "but only {} are specified: {}"
                                .format(nlay, nspecified, self.from_source_model_layers))
            return self.from_source_model_layers

    @property
    def kstpkper(self):
        """Currently this class is only intended to produce a single 3D array
        for a given timestep/stress period. Find the last timestep
        associated with the period argument (to __init__) and return the
        a (kstp, kper) tuple for getting the binary data.
        """
        if self.filename.endswith('hds'):
            bfobj = bf.HeadFile(self.filename)
            kstpkpers = bfobj.get_kstpkper()
            for kstp, kper in kstpkpers:
                if kper == self.stress_period:
                    kstpkper = kstp, kper
                if kper > self.stress_period:
                    break
            return kstpkper

        elif self.filename[:-4] in {'.cbb', '.cbc'}:
            raise NotImplementedError('Cell Budget files not supported yet.')

    def get_data(self, **kwargs):
        """Get array data from binary file for a single time;
        regrid from source model to dest model and transfer layer
        data from source model to dest model based on from_source_model_layers
        argument to class.

        Parameters
        ----------
        kwargs : keyword arguments to flopy.utils.binaryfile.HeadFile

        Returns
        -------
        data : dict
            Dictionary of 2D arrays keyed by destination model layer.
        """

        if self.filename.endswith('hds'):
            bfobj = bf.HeadFile(self.filename)
            self.source_array = bfobj.get_data(kstpkper=self.kstpkper)

        elif self.filename[:-4] in {'.cbb', '.cbc'}:
            raise NotImplementedError('Cell Budget files not supported yet.')

        data = {}
        # interpolate by layer, using the layer mapping specified by the user
        if self.from_source_model_layers is not None:
            for dest_k, source_k in self.dest_source_layer_mapping.items():

                # destination model layers copied from source model layers
                if source_k <= 0:
                    arr = self.source_array[0]
                elif np.round(source_k, 4) in range(self.source_array.shape[0]):
                    source_k = int(np.round(source_k, 4))
                    arr = self.source_array[source_k]
                # destination model layers that are a weighted average
                # of consecutive source model layers
                # TODO: add transmissivity-based weighting if upw exists
                else:
                    weight0 = source_k - np.floor(source_k)
                    source_k0 = int(np.floor(source_k))
                    # first layer in the average can't be negative
                    source_k0 = 0 if source_k0 < 0 else source_k0
                    source_k1 = int(np.ceil(source_k))
                    arr = weighted_average_between_layers(self.source_array[source_k0],
                                                          self.source_array[source_k1],
                                                          weight0=weight0)
                # interpolate from source model using source model grid
                # otherwise assume the grids are the same
                if self.source_modelgrid is not None:
                    # exclude invalid values in interpolation from parent model
                    mask = self._source_grid_mask & (arr > self.vmin) & (arr < self.vmax)

                    arr = self.regrid_from_source_model(arr,
                                                        mask=mask,
                                                        method='linear')

                assert arr.shape == self.dest_modelgrid.shape[1:]
                data[dest_k] = arr * self.mult * self.unit_conversion
        # general 3D interpolation based on the location of parent and inset model cells
        else:
            # tile the mask to nlay x nrow x ncol
            in_window = np.tile(self._source_grid_mask, (self.source_modelgrid.nlay, 1, 1))
            valid = (self.source_array > self.vmin) & (self.source_array < self.vmax)
            mask = valid & in_window
            heads = regrid3d(self.source_array, self.source_modelgrid, self.dest_modelgrid,
                             mask1=mask, method='linear')
            data = {k: heads2d for k, heads2d in enumerate(heads)}

        self.data = data
        return data


class MFArrayData(SourceData):
    """Subclass for handling array-based source data that can
    be scalars, lists of scalars, array data or filepath(s) to arrays on
    same model grid."""
    def __init__(self, variable, filenames=None, values=None, length_units='unknown', time_units='unknown',
                 dest_model=None, vmin=-1e30, vmax=1e30, dtype=float, datatype=None,
                 multiplier=1., **kwargs):

        SourceData.__init__(self, filenames=filenames, length_units=length_units, time_units=time_units,
                            dest_model=dest_model)

        self.variable = variable
        self.values = values
        self.vmin = vmin
        self.vmax = vmax
        self.mult = multiplier
        self.dest_modelgrid = getattr(self.dest_model, 'modelgrid', None)
        self.dtype = dtype
        self.datatype = datatype
        self.data = {}
        assert True

    def get_data(self):
        data = {}

        # number of layers in dest. array
        # (array3d dtype is nlay, nrow, ncol)
        if self.datatype == 'array3d':
            nk = self.dest_model.nlay
        elif self.datatype == 'transient2d':
            nk = self.dest_model.nper
        else:
            nk = 1

        # tile 2D array to 3D
        if isinstance(self.values, np.ndarray) and \
                len(self.values.shape) == 2:
            self.values = np.tile(self.values, (nk, 1, 1))

        # convert to dict
        if isinstance(self.values, str) or np.isscalar(self.values):
            self.values = {k: self.values for k in range(nk)}
        elif isinstance(self.values, list) or isinstance(self.values, np.ndarray):
            self.values = {i: val for i, val in enumerate(self.values)}
        for i, val in self.values.items():
            if isinstance(val, dict) and 'filename' in val.keys():
                val = val['filename']
            if isinstance(val, str):
                abspath = os.path.normpath(os.path.join(self.dest_model.model_ws, val))
                arr = np.loadtxt(abspath)
            elif np.isscalar(val):
                arr = np.ones(self.dest_modelgrid.shape[1:]) * val
            else:
                arr = val
            assert arr.shape == self.dest_modelgrid.shape[1:]
            data[i] = arr * self.mult * self.unit_conversion

        if self.datatype == 'array3d':
            if len(data) != self.dest_model.nlay:
                raise Exception("Variable should have {} layers "
                                "but only {} are specified: {}"
                                .format(self.dest_model.nlay,
                                        len(data),
                                        self.values))
        self.data = data
        return data


class TabularSourceData(SourceData):
    """Subclass for handling tabular source data."""

    def __init__(self, filenames, id_column=None, include_ids=None,
                 data_column=None, sort_by=None,
                 length_units='unknown', time_units='unknown', volume_units=None,
                 column_mappings=None,
                 dest_model=None):
        SourceData.__init__(self, filenames=filenames,
                            length_units=length_units, time_units=time_units,
                            volume_units=volume_units,
                            dest_model=dest_model)

        self.id_column = id_column
        self.include_ids = include_ids
        self.data_column = data_column
        self.column_mappings = column_mappings
        self.sort_by = sort_by

    def get_data(self):

        dfs = []
        for i, f in self.filenames.items():
            if f.endswith('.shp') or f.endswith('.dbf'):
                # implement automatic reprojection in gis-utils
                # maintaining backwards compatibility
                kwargs = {'dest_crs': self.dest_model.modelgrid.crs}
                kwargs = get_input_arguments(kwargs, shp2df)
                df = shp2df(f, **kwargs)
                df.columns = [c.lower() for c in df.columns]

            elif f.endswith('.csv'):
                df = pd.read_csv(f)

            dfs.append(df)

        df = pd.concat(dfs)
        if self.id_column is not None:
            df.index = df[self.id_column]
        if self.include_ids is not None:
            df = df.loc[self.include_ids]
        if self.data_column is not None:
            df[self.data_column] *= self.unit_conversion
        if self.sort_by is not None:
            df.sort_values(by=self.sort_by, inplace=True)

        # rename any columns specified in config file to required names
        if self.column_mappings is not None:
            df.rename(columns=self.column_mappings, inplace=True)
        #df.columns = [c.lower() for c in df.columns]

        # drop any extra unnamed columns from accidental saving of the index on to_csv
        drop_columns = [c for c in df.columns if 'unnamed' in c]
        df.drop(drop_columns, axis=1, inplace=True)
        return df.reset_index(drop=True)


class TransientTabularSourceData(SourceData, TransientSourceDataMixin):
    """Subclass for handling tabular source data that
    represents a time series."""

    def __init__(self, filenames, data_columns, datetime_column, id_column,
                 x_col='x', y_col='y', end_datetime_column=None, period_stats={0: 'mean'},
                 length_units='unknown', time_units='unknown', volume_units=None,
                 column_mappings=None, category_column=None,
                 dest_model=None, resolve_duplicates_with='raise error', **kwargs):
        SourceData.__init__(self, filenames=filenames,
                            length_units=length_units, time_units=time_units,
                            volume_units=volume_units,
                            dest_model=dest_model)
        TransientSourceDataMixin.__init__(self, period_stats=period_stats, dest_model=dest_model)
        if isinstance(data_columns, str):
            data_columns = [data_columns]
        self.data_columns = data_columns
        self.datetime_column = datetime_column
        self.end_datetime_column = end_datetime_column
        self.id_column = id_column
        self.column_mappings = column_mappings
        self.category_column = category_column
        self.time_col = datetime_column
        self.x_col = x_col
        self.y_col = y_col
        self.resolve_duplicates_with = resolve_duplicates_with

    def get_data(self):

        # aggregate the data from multiple files
        dfs = []
        for i, f in self.filenames.items():
            if str(f).endswith('.shp') or str(f).endswith('.dbf'):
                # implement automatic reprojection in gis-utils
                # maintaining backwards compatibility
                kwargs = {'dest_crs': self.dest_model.modelgrid.crs}
                kwargs = get_input_arguments(kwargs, shp2df)
                df = shp2df(f, **kwargs)
            elif str(f).endswith('.csv'):
                df = pd.read_csv(f)
            else:
                raise ValueError("Unsupported file type: '{}', for {}".format(f[:-4], f))
            dfs.append(df)
        df = pd.concat(dfs)
        df.index = pd.to_datetime(df[self.datetime_column])

        # convert IDs to strings if any were read in (resulting in object dtype)
        if df[self.id_column].dtype == object:
            df[self.id_column] = df[self.id_column].astype(str)

        # rename any columns specified in config file to required names
        if self.column_mappings is not None:
            df.rename(columns=self.column_mappings, inplace=True)

        # cull data to model bounds
        has_locations = False
        if 'geometry' not in df.columns or isinstance(df.geometry.iloc[0], str):
            if self.x_col in df.columns and self.y_col in df.columns:
                df['geometry'] = [Point(x, y) for x, y in zip(df[self.x_col], df[self.y_col])]
                has_locations = True
        else:
            has_locations = True
        if has_locations:
            within = [g.within(self.dest_model.bbox) for g in df.geometry]
            df = df.loc[within]
        if self.end_datetime_column is None:
            msg = '\n'.join(self.filenames.values()) + ':\n'
            msg += ('Transient tabular time-series with no end_datetime_column specified.\n'
                    'Data on time intervals longer than the model stress periods may not be\n'
                    'upsampled correctly, as dates in the datetime_column are used for '
                    'intersection with model stress periods.')
            warnings.warn(msg)
        period_data = []
        for kper, period_stat in self.period_stats.items():
            if period_stat is None:
                continue
            aggregated = aggregate_dataframe_to_stress_period(df, id_column=self.id_column,
                                                              datetime_column=self.datetime_column,
                                                              end_datetime_column=self.end_datetime_column,
                                                              data_column=self.data_columns,
                                                              category_column=self.category_column,
                                                              resolve_duplicates_with=self.resolve_duplicates_with,
                                                              **period_stat)
            aggregated['per'] = kper
            period_data.append(aggregated)
        dfm = pd.concat(period_data)

        if self.data_columns is not None:
            for col in self.data_columns:
                dfm[col] *= self.unit_conversion
        dfm.sort_values(by=['per', self.id_column], inplace=True)

        # drop any extra unnamed columns from accidental saving of the index on to_csv
        drop_columns = [c for c in dfm.columns if 'unnamed' in c]
        dfm.drop(drop_columns, axis=1, inplace=True)

        # map x, y locations to modelgrid
        if has_locations:
            i, j = get_ij(self.dest_model.modelgrid,
                          dfm[self.x_col].values, dfm[self.y_col].values)
            dfm['i'] = i
            dfm['j'] = j
        return dfm


def setup_array(model, package, var, data=None,
                vmin=-1e30, vmax=1e30, datatype=None,
                source_model=None, source_package=None,
                write_fmt='%.6e', write_nodata=None,
                **kwargs):
    """Todo: this method really needs to be cleaned up and maybe refactored

    Parameters
    ----------
    model :
    package :
    var :
    data :
    vmin :
    vmax :
    source_model :
    source_package :
    write_fmt :
    write_nodata :
    kwargs :

    Returns
    -------

    """

    # based on MODFLOW version
    # get direct model input if it is specified
    # configure external file handling
    if model.version == 'mf6':
        if data is None:
            data = model.cfg[package].get('griddata', {})
            if isinstance(data, dict):
                data = data.get(var)
        external_files_key = 'external_files'
    else:
        if data is None:
            data = model.cfg[package].get(var)
        external_files_key = 'intermediate_data'

    # get any source_data input
    # if default_source_data: True in parent model options
    # default to source_data from parent model
    cfg = model.cfg[package].get('source_data')
    if cfg is None and model.cfg['parent'].get('default_source_data')\
        and not (model._is_lgr and var == 'idomain'):
        cfg = {var: 'from_parent'}

    # data specified directly
    sd = None
    if data is not None:
        sd = MFArrayData(variable=var, values=data, dest_model=model,
                         datatype=datatype,
                         vmin=vmin, vmax=vmax,
                         **kwargs)

    # data specified as source_data
    elif cfg is not None and var in cfg:

        from_model = False

        # get the source data block
        # determine if source data is from another model
        source_data_input = cfg.get(var)
        if source_data_input == 'from_parent':
            from_model_keys = [source_data_input]
            from_model = True
        elif isinstance(source_data_input, dict):
            from_model_keys = [k for k in source_data_input.keys() if 'from_' in k]
            from_model = True if len(from_model_keys) > 0 else False
            # give source_data_input priority over kwargs, which are assumed to be defaults
            kwargs = {k: v for k, v in kwargs.items() if k not in source_data_input}

        # data from files
        if not from_model:
            ext = get_source_data_file_ext(source_data_input, package, var)

            if datatype == 'transient2d' and ext == '.nc':
                sd = NetCDFSourceData.from_config(source_data_input,
                                                  datatype=datatype,
                                                  dest_model=model,
                                                  vmin=vmin, vmax=vmax,
                                                  **kwargs
                                                  )
            elif datatype == 'transient2d':
                sd = TransientArraySourceData.from_config(source_data_input,
                                                          variable=var,
                                                          datatype=datatype,
                                                  dest_model=model,
                                                  vmin=vmin, vmax=vmax,
                                                  **kwargs
                                                  )
            else:
                # TODO: files option doesn't support interpolation between top and botm[0]
                sd = ArraySourceData.from_config(source_data_input,
                                                 datatype=datatype,
                                                 variable=var,
                                                 dest_model=model,
                                                 vmin=vmin, vmax=vmax,
                                                 **kwargs)

        # data regridded from a source model
        elif from_model:

            # Determine mapping between source model and dest model
            binary_file = False
            filenames = None
            # read source model data from specified layers
            # to specified layers in dest model
            if isinstance(source_data_input, dict):
                key = from_model_keys[0]
                from_source_model_layers = source_data_input[key].copy() #cfg[var][key].copy()
                # instead of package input arrays, source model data is from a binary file
                if 'binaryfile' in from_source_model_layers:
                    binary_file = True
                    filename = from_source_model_layers.pop('binaryfile')
            # otherwise, read all source data to corresponding layers in dest model
            elif source_data_input == 'from_parent':
                key = source_data_input
                from_source_model_layers = None
            modelname = key.split('_')[1]

            # TODO: could generalize this to allow for more than one source model
            if modelname == 'parent':
                source_model = model.parent

            # for getting data from a different package in the source model
            source_variable = get_variable_name(var, source_model.version)
            source_package = get_variable_package_name(var, source_model.version, package)
            source_package_instance = getattr(source_model, source_package, None)
            txt = 'No variable {} in source model {}, {} package. Skipping...'.format(source_variable,
                                                                                      source_model.name,
                                                                                      source_package)
            # traps against variables that might not exist
            if var in ['ss', 'sy'] and source_model.perioddata.steady.all():
                return
            if source_package_instance is not None:
                source_variable_exists = getattr(source_package_instance, source_variable, False)
                if not source_variable_exists:
                    print(txt)
                    return
            else:
                print(txt)
                return

            # data from parent model MODFLOW binary output
            if binary_file:
                sd = MFBinaryArraySourceData(variable=source_variable, filename=filename,
                                             datatype=datatype,
                                             dest_model=model,
                                             source_modelgrid=source_model.modelgrid,
                                             from_source_model_layers=from_source_model_layers,
                                             length_units=model.cfg[modelname]['length_units'],
                                             time_units=model.cfg[modelname]['time_units'],
                                             vmin=vmin, vmax=vmax,
                                             **kwargs)

            # data read from Flopy instance of parent model
            elif source_model is not None:
                # the botm array has to be handled differently
                # because dest. layers may be interpolated between
                # model top and first botm
                source_array_includes_model_top = False
                if source_variable == 'botm' and \
                        from_source_model_layers is not None and \
                        from_source_model_layers[0] < 0:
                    nlay, nrow, ncol = source_model.dis.botm.array.shape
                    source_array = np.zeros((nlay+1, nrow, ncol))
                    source_array[0] = source_model.dis.top.array
                    source_array[1:] = source_model.dis.botm.array
                    from_source_model_layers = {k: v+1 for k, v in from_source_model_layers.items()}
                    source_array_includes_model_top = True
                else:
                    source_array = getattr(source_model, source_package).__dict__[source_variable].array

                if datatype == 'transient2d':
                    sd = TransientArraySourceData(variable=source_variable, filenames=filenames,
                                                  datatype=datatype,
                                                  dest_model=model,
                                                  source_modelgrid=source_model.modelgrid,
                                                  source_array=source_array,
                                                  length_units=model.cfg[modelname]['length_units'],
                                                  time_units=model.cfg[modelname]['time_units'],
                                                  vmin=vmin, vmax=vmax,
                                                  **kwargs)
                else:
                    sd = ArraySourceData(variable=source_variable, filenames=filenames,
                                         datatype=datatype,
                                         dest_model=model,
                                         source_modelgrid=source_model.modelgrid,
                                         source_array=source_array,
                                         from_source_model_layers=from_source_model_layers,
                                         source_array_includes_model_top=source_array_includes_model_top,
                                         length_units=model.cfg[modelname]['length_units'],
                                         time_units=model.cfg[modelname]['time_units'],
                                         vmin=vmin, vmax=vmax,
                                         **kwargs)
            if var == 'vka':
                model.cfg['upw']['layvka'] = getattr(source_model, source_package).layvka.array[0]
        else:
            raise Exception("No source data found for {} package: {}".format(package, var))

    # default for idomain if no other information provided
    # rest of the setup is handled by idomain property
    elif var in ['idomain', 'ibound']:
        sd = MFArrayData(variable=var, values=1, datatype=datatype, dest_model=model, **kwargs)

    # default to model top for starting heads
    elif var == 'strt':
        sd = MFArrayData(variable=var,
                         values=[model.dis.top.array] * model.nlay,
                         datatype=datatype,
                         dest_model=model, **kwargs)

    # no data were specified, or input not recognized
    elif sd is None:
        print('No data were specified for {} package, variable {}'.format(package, var))
        return

    data = sd.get_data()

    # special handling of some variables
    # (for lakes)
    simulate_high_k_lakes = model.cfg['high_k_lakes']['simulate_high_k_lakes']
    if var == 'botm':
        if 'lak' in model.cfg['model']['packages'] and not model._load and\
            np.sum(model.lake_bathymetry) != 0:
            # only execute this code if building the model (not loading)
            # and if lake bathymetry was supplied
            # (bathymetry is loaded from the configuration file input
            #  when the model.lake_bathymetry property is called for the first time;
            #  see MFsetupMixin._set_lake_bathymetry)
            bathy = model.lake_bathymetry

            # save a copy of original top elevations
            # (prior to adjustment for lake bathymetry)
            # name of the copy:
            original_top_file = Path(model.external_path,
                                    f"{model.name}_{model.cfg[package]['top_filename_fmt']}.original")

            try:
                top = model.load_array(original_top_file)
                original_top_load_fail = False
            except:
                original_top_load_fail = True

            # if the copy doesn't exist
            # (or if the existing file is invalid), make it
            if original_top_load_fail:
                # if remake_top is False, however,
                # there may be no preexisting top file to copy
                # first check for a preexisting top file
                # get the path and add to intermediate files dict if it's not in there
                if 'top' not in model.cfg['intermediate_data']:
                    model.setup_external_filepaths('dis', 'top',
                                                model.cfg['dis']['top_filename_fmt'])
                existing_model_top_file = Path(model.cfg['intermediate_data']['top'][0])
                if not existing_model_top_file.exists():
                    raise ValueError((f"Model top text array file {existing_model_top_file} doesn't exist.\n"
                                    f"If remake_top is False in the dis configuration block, "
                                    f"{existing_model_top_file} needs to have been made previously."))
                # copy the preexisting top file
                shutil.copy(model.cfg['intermediate_data']['top'][0],
                            original_top_file)
                top = model.load_array(original_top_file)
            if model.version == 'mf6':
                # reset the model top to the lake bottom
                top[bathy != 0] -= bathy[bathy != 0]

                # update the top in the model
                # todo: refactor this code to be consolidated with _set_idomain and setup_dis
                model._setup_array('dis', 'top',
                        data={0: top},
                        datatype='array2d', resample_method='linear',
                        write_fmt='%.2f', dtype=float)
                if hasattr(model, 'dis') and model.dis is not None:
                    if model.version == 'mf6':
                        model.dis.top = model.cfg['dis']['griddata']['top']
                    else:
                        model.dis.top = model.cfg['dis']['top'][0]

                # write the top array again
                top_filepath = model.setup_external_filepaths(package, 'top',
                                                            model.cfg[package]['top_filename_fmt'])[0]
                save_array(top_filepath, top,
                        nodata=write_nodata,
                        fmt=write_fmt)
        # if loading the model; use the model top that was just loaded in
        else:
            top_filename = model.cfg['dis']['griddata'].get('top')
            if top_filename is None:
                model.setup_external_filepaths('dis', 'top',
                                                model.cfg['dis']['top_filename_fmt'])
            if model.version == 'mf6':
                top_filename = model.cfg['dis']['griddata']['top'][0]['filename']
            else:
                top_filename = model.cfg['dis']['top'][0]
            top = model.load_array(top_filename)

        # fill missing layers if any
        if len(data) < model.nlay:
            all_surfaces = np.zeros((model.nlay + 1, model.nrow, model.ncol), dtype=float) * np.nan
            # for LGR models, populate any missing bottom layer(s)
            # from the refined parent grid
            # this allows for the LGR model to be
            # bounded on the bottom by a sub-divided parent layer surface
            last_specified_layer = max(data.keys())
            if model._is_lgr and last_specified_layer < (model.nlay - 1):
                #for k in range(last_specified_layer + 1, model.nlay):
                #    all_surfaces[k+1:] = model.parent.lgr[model.name].botm[k]
                all_surfaces[model.nlay] = model.parent.lgr[model.name].botm[model.nlay -1]
            all_surfaces[0] = top
            for k, botm in data.items():
                all_surfaces[k + 1] = botm
            all_surfaces = fill_empty_layers(all_surfaces)
            botm = all_surfaces[1:]
        else:
            botm = np.stack([data[i] for i in range(len(data))])

        # adjust layer botms to lake bathymetry (if any)
        # set layer bottom at lake cells to the botm of the lake in that layer
        # move layer bottoms down, except for the first layer (move the model top down)
        #botm[botm > top] = np.array([top]*5)[botm > top]
        #i, j = np.where(bathy != 0)
        #layers = get_layer(botm, i, j, lake_botm_elevations)
        #layers[layers > 0] -= 1
        ## include any overlying layers
        #deact_lays = [list(range(k)) for k in layers]
        #for ks, ci, cj in zip(deact_lays, i, j):
        #    for ck in ks:
        #        botm[ck, ci, cj] = np.nan

        # the model top was reset above to any lake bottoms
        # (defined by bathymetry)
        # deactivate any bottom elevations above or equal to new top
        # (decativate cell bottoms above or equal to the lake bottom)
        # then deactivate these zero-thickness cells
        # (so they won't get expanded again by fix_model_layer_conflicts)
        # only do this for mf6, where pinched out cells are allowed
        min_thickness = model.cfg['dis'].get('minimum_layer_thickness', 1)
        if model.version == 'mf6':
            botm[botm >= (top - min_thickness)] = np.nan

        #for k, kbotm in enumerate(botm):
        #    inlayer = lake_botm_elevations > kbotm[bathy != 0]
        #    if not np.any(inlayer):
        #        continue
        #    botm[k][bathy != 0][inlayer] = lake_botm_elevations[inlayer]

        # fix any layering conflicts and save out botm files
        # only adjust layer elevations if we want to keep thin cells
        # (instead of making them inactive)
        if not model._drop_thin_cells:
            botm = fix_model_layer_conflicts(top, botm,
                                            minimum_thickness=min_thickness)
            isvalid = verify_minimum_layer_thickness(top, botm,
                                                    np.ones(botm.shape, dtype=int),
                                                    min_thickness)
            if not isvalid:
                raise Exception('Model layers less than {} {} thickness'.format(min_thickness,
                                                                            model.length_units))
        # fill any nan values that are above or below active cells to avoid cell thickness errors
        botm = fill_cells_vertically(top, botm)

        data = {i: arr for i, arr in enumerate(botm)}

        # special case of LGR models
        # with bottom connections to underlying parent cells
        if model.version == 'mf6':
            # (if model is an lgr inset model)
            if model._is_lgr:
                # regardless of what is specified for inset model bottom
                # use top elevations of underlying parent model cells
                nlay = model.cfg['dis']['dimensions']['nlay']
                lgr = model.parent.lgr[model.name]  # Flopy Lgr inst.
                n_refined = (np.array(lgr.ncppl) > 0).astype(int).sum()
                # check if LGR model has connections to underlying parent cells
                if (n_refined < model.parent.modelgrid.nlay):
                    # use the parent model bottoms
                    # mapped to the inset model grid by the Flopy Lgr util
                    #data[nlay-1] = lgr.botm[nlay-1]
                    # set the inset and parent model botms
                    # to the mean elevations of the inset cells
                    # connecting to each parent cell
                    ncpp = lgr.ncpp
                    nrowp = int(data[nlay-1].shape[0]/ncpp)
                    ncolp = int(data[nlay-1].shape[1]/ncpp)
                    lgr_inset_botm = data[nlay-1]
                    if any(np.isnan(lgr_inset_botm.flat)):
                        raise ValueError(
                            "Nan values in LGR inset model bottom; "
                            "can't use this to make the top of the parent model."
                            )
                    # average each nccp x nccp block of inset model cells
                    # nccp = number of child cells per parent cell
                    new_parent_top = np.reshape(lgr_inset_botm,
                                   (nrowp, ncpp, ncolp, ncpp)).mean(axis=(1, 3))
                    n_parent_lgr_layers = np.sum(np.array(lgr.ncppl) > 0)
                    # remap averages back to the inset model shape
                    # and assign to inset model bottom
                    data[nlay-1] = np.repeat(np.repeat(new_parent_top, ncpp, axis=0),
                                             ncpp, axis=1)
                    # set the parent model top in this area to be the same
                    lgr_area = model.parent.lgr[model.name].idomain == 0
                    model.parent.dis.top[lgr_area[0]] = new_parent_top.ravel()
                    # set parent model layers in LGR area to zero-thickness
                    new_parent_botm = model.parent.dis.botm.array.copy()
                    for k in range(n_parent_lgr_layers):
                        new_parent_botm[k][lgr_area[0]] = new_parent_top.ravel()
                    model.parent.dis.botm = new_parent_botm
                    model.parent._update_top_botm_external_files()

    elif var in ['rech', 'recharge']:
        if simulate_high_k_lakes:
            for per in range(model.nper):
                if per == 0 and per not in data:
                    raise KeyError("No recharge input specified for first stress period.")
                if per in data:
                    # assign high-k lake recharge for stress period
                    # only assign if precip and open water evaporation data were read
                    # (otherwise keep original values in recharge array)
                    last_data_array = data[per].copy()
                    if model.high_k_lake_recharge is not None:
                        data[per][model.isbc[0] == 2] = model.high_k_lake_recharge[per]
                    # zero-values to lak package lakes
                    data[per][model.isbc[0] == 1] = 0.
                else:
                    if model.high_k_lake_recharge is not None:
                        # start with the last period with recharge data; update the high-k lake recharge
                        last_data_array[model.isbc[0] == 2] = model.high_k_lake_recharge[per]
                    last_data_array[model.isbc[0] == 1] = 0.
                    # assign to current per
                    data[per] = last_data_array
        if model.lgr:
            for per, rech_array in data.items():
                rech_array[model._lgr_idomain2d == 0] = 0
    elif var in ['hk', 'k'] and simulate_high_k_lakes:
        for i, arr in data.items():
            data[i][model.isbc[i] == 2] = model.cfg['high_k_lakes']['high_k_value']
    elif var == 'sy' and simulate_high_k_lakes:
        for i, arr in data.items():
            data[i][model.isbc[i] == 2] = model.cfg['high_k_lakes']['sy']
    elif var == 'ss' and simulate_high_k_lakes:
        for i, arr in data.items():
            data[i][model.isbc[i] == 2] = model.cfg['high_k_lakes']['ss']
    # intermediate data
    # set paths to intermediate files and external files
    filepaths = model.setup_external_filepaths(package, var,
                                               model.cfg[package]['{}_filename_fmt'.format(var)],
                                               file_numbers=list(data.keys()))
    # write out array data to intermediate files
    # assign lake recharge values (water balance surplus) for any high-K lakes
    if write_nodata is None:
        write_nodata = model._nodata_value
    for i, arr in data.items():
        save_array(filepaths[i], arr,
                nodata=write_nodata,
                fmt=write_fmt)
        # still write intermediate files for MODFLOW-6
        # even though input and output filepaths are same
        if model.version == 'mf6':
            src = filepaths[i]['filename']
            dst = model.cfg['intermediate_data'][var][i]
            shutil.copy(src, dst)


def get_source_data_file_ext(cfg_data, package, var):
    if 'filenames' in cfg_data:
        if isinstance(cfg_data['filenames'], dict):
            filename = list(cfg_data['filenames'].values())[0]
        elif isinstance(cfg_data['filenames'], list):
            filename = cfg_data['filenames'][0]
    elif 'filename' in cfg_data:
        filename = cfg_data['filename']
    else:
        raise ValueError('Source_data for {}: {} needs one or more filenames!'.format(package, var))
    _, ext = os.path.splitext(filename)
    return ext


def transient2d_to_xarray(data, x=None, y=None, time=None):
    if x is None:
        x = np.arange(data.shape[2])
    if y is None:
        y = np.arange(data.shape[1])[::-1]
    if time is None:
        time = np.arange(data.shape[0])
    da = xr.DataArray(data=data,
                      coords={"x": x,
                              "y": y,
                              "time": time},
                      dims=["x", "y", "time"])
    return da
