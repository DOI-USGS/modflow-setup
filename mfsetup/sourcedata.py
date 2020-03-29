import os
import shutil
import numbers
import warnings
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point
import pandas as pd
import xarray as xr
from flopy.utils import binaryfile as bf
from mfsetup.discretization import weighted_average_between_layers
from mfsetup.tdis import aggregate_dataframe_to_stress_period, aggregate_xarray_to_stress_period
from .fileio import save_array
from .discretization import (fix_model_layer_conflicts, verify_minimum_layer_thickness,
                             fill_empty_layers, fill_cells_vertically, populate_values)
from gisutils import (get_values_at_points, shp2df)
from .grid import get_ij, rasterize
from .interpolate import get_source_dest_model_xys, interp_weights, interpolate, regrid
from .mf5to6 import get_variable_package_name, get_variable_name
from .units import (convert_length_units, convert_time_units, convert_volume_units)
from .utils import get_input_arguments

renames = {'mult': 'multiplier',
           'elevation_units': 'length_units',
           'from_parent': 'from_source_model_layers',
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
    def __init__(self, filenames=None, values=None, length_units='unknown',
                 time_units='unknown',
                 area_units=None, volume_units=None,
                 datatype=None,
                 dest_model=None):
        """

        """
        self.filenames = filenames
        self.values = values
        self.length_units = length_units
        self.area_units = area_units
        self.volume_units = volume_units
        self.time_units = time_units
        self.datatype = datatype
        self.dest_model = dest_model
        self.set_filenames(filenames)

    @property
    def unit_conversion(self):
        return self.length_unit_conversion * self.time_unit_conversion

    @property
    def length_unit_conversion(self):
        # data are lengths
        mult = convert_length_units(self.length_units,
                                    getattr(self.dest_model, 'length_units', 'unknown'))

        # data are areas
        if self.area_units is not None:
            raise NotImplemented('Conversion of area units.')

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
                try:
                    path = os.path.join(self.dest_model._config_path, f)
                except:
                    j=2
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
                 from_source_model_layers=None, datatype=None,
                 id_column=None, include_ids=None, column_mappings=None,
                 resample_method='nearest',
                 vmin=-1e30, vmax=1e30, dtype=float,
                 multiplier=1.):

        SourceData.__init__(self, filenames=filenames, values=values,
                            length_units=length_units, time_units=time_units,
                            datatype=datatype,
                            dest_model=dest_model)

        self.variable = variable
        self.source_modelgrid = source_modelgrid
        self._source_mask = None
        if from_source_model_layers == {}:
            from_source_model_layers = None
        self.from_source_model_layers = from_source_model_layers
        self.source_array = None
        if source_array is not None:
            if len(source_array.shape) == 2:
                source_array = source_array.reshape(1, *source_array.shape)
            self.source_array = source_array
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
            if f.endswith(".asc") or f.endswith(".tif"):
                if self.resample_method != 'nearest':
                    warnings.warn('{}: resample method {} not implemented; '
                                  'falling back to nearest'.format(self.variable,
                                                                   self.resample_method))
                arr = get_values_at_points(f,
                                           self.dest_model.modelgrid.xcellcenters.ravel(),
                                           self.dest_model.modelgrid.ycellcenters.ravel())
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
            if self.datatype == 'array3d' and i < (self.dest_model.nlay - 1):
                for j in range(i, self.dest_model.nlay):
                    data[j] = data[i]

        # regrid source data from another model
        elif self.source_array is not None:

            for dest_k, source_k in self.dest_source_layer_mapping.items():
                if source_k >= self.source_array.shape[0]:
                    continue
                # destination model layers copied from source model layers
                # if source_array has an extra layer, assume layer 0 is the model top
                # (only included for weighted average)
                # could use a better approach
                if np.round(source_k, 4) in range(self.source_array.shape[0]):
                    if self.source_array.shape[0] - self.dest_model.nlay == 1:
                        source_k +=1
                    source_k = int(np.round(source_k, 4))
                    arr = self.source_array[source_k]
                # destination model layers that are a weighted average
                # of consecutive source model layers
                else:
                    weight0 = source_k - np.floor(source_k)
                    source_k0 = int(np.floor(source_k))
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
                                                        method='linear')

                assert regridded.shape == self.dest_modelgrid.shape[1:]
                data[dest_k] = regridded * self.mult * self.unit_conversion

        # no files or source array provided
        else:
            raise ValueError("No files or source model grid provided.")

        self.data = data
        return data


class TransientArraySourceData(ArraySourceData):
    def __init__(self, filenames, variable, period_stats=None,
                 length_units='unknown', time_units='days',
                 dest_model=None, source_modelgrid=None, source_array=None,
                 from_source_model_layers=None, datatype=None,
                 resample_method='nearest', vmin=-1e30, vmax=1e30
                 ):

        ArraySourceData.__init__(self, variable=None, filenames=filenames,
                                 length_units=length_units, time_units=time_units,
                                 dest_model=dest_model, source_modelgrid=source_modelgrid,
                                 source_array=source_array,
                                 from_source_model_layers=from_source_model_layers,
                                 datatype=datatype,
                                 resample_method=resample_method, vmin=vmin, vmax=vmax)

        self.variable = variable
        self.period_stats = period_stats
        self.resample_method = resample_method
        self.dest_model = dest_model

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
            source_data = self.source_array
            regrid = True

        # cast data to an xarray DataArray for time-sliceing
        # TODO: implement temporal resampling from source model
        # would follow logic of netcdf files, but trickier because steady-state periods need to be handled
        #da = transient2d_to_xarray(data, time)

        # for now, just assume one-to-one correspondance
        # between source and dest model stress periods
        results = {}
        for kper in range(self.dest_model.nper):
            if kper >= len(source_data):
                data = source_data[-1]
            else:
                data = source_data[kper]
            if regrid:
                # sample the data onto the model grid
                resampled = self.regrid_from_source_model(data,
                                                          method=self.resample_method)
            else:
                resampled = data
            # reshape results to model grid
            period_mean2d = resampled.reshape(self.dest_model.nrow,
                                              self.dest_model.ncol)
            results[kper] = period_mean2d * self.unit_conversion
        self.data = results
        return results


class NetCDFSourceData(ArraySourceData):
    def __init__(self, filenames, variable, period_stats,
                 length_units='unknown', time_units='days',
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

        if isinstance(filenames, dict):
            filenames = list(filenames.values())
        if isinstance(filenames, list):
            if len(filenames) > 1:
                raise NotImplementedError("Multiple NetCDF files not supported.")
            self.filename = filenames[0]
        else:
            self.filename = filenames
        self.variable = variable
        self.period_stats = period_stats
        self.resample_method = resample_method
        self.dest_model = dest_model
        self.time_col = 'time'

        # set xy value arrays for source and dest. grids
        with xr.open_dataset(self.filename) as ds:
            x1, y1 = np.meshgrid(ds.x.values, ds.y.values)
            x1 = x1.ravel()
            y1 = y1.ravel()
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
        # TODO: make this general for using with lists of files or other input by stress period
        starttimes = self.dest_model.perioddata['start_datetime']
        endtimes = self.dest_model.perioddata['end_datetime']
        results = {}
        current_stat = None
        for kper, (start, end) in enumerate(zip(starttimes, endtimes)):
            period_stat = self.period_stats.get(kper, current_stat)
            current_stat = period_stat
            aggregated = aggregate_xarray_to_stress_period(data,
                                                           start_datetime=start,
                                                           end_datetime=end,
                                                           period_stat=period_stat,
                                                           datetime_column=self.time_col)

            # sample the data onto the model grid
            resampled = self.regrid_from_source(aggregated,
                                                method=self.resample_method)

            # reshape results to model grid
            period_mean2d = resampled.reshape(self.dest_model.nrow,
                                              self.dest_model.ncol)
            results[kper] = period_mean2d * self.unit_conversion
        self.data = results
        return results


class MFBinaryArraySourceData(ArraySourceData):
    """Subclass for handling MODFLOW binary array data
    that may come from another model."""
    def __init__(self, variable, filename=None,
                 length_units='unknown', time_units='unknown',
                 dest_model=None, source_modelgrid=None,
                 from_source_model_layers=None, datatype='transient3d',
                 resample_method='nearest', vmin=-1e30, vmax=1e30
                 ):

        ArraySourceData.__init__(self, variable=variable,
                                 length_units=length_units, time_units=time_units,
                                 dest_model=dest_model, source_modelgrid=source_modelgrid,
                                 from_source_model_layers=from_source_model_layers,
                                 datatype=datatype,
                                 resample_method=resample_method, vmin=vmin, vmax=vmax)

        self.filename = filename

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
            self.source_array = bfobj.get_data(**kwargs)

        elif self.filename[:-4] in {'.cbb', '.cbc'}:
            raise NotImplementedError('Cell Budget files not supported yet.')

        data = {}
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

        # convert to dict
        if isinstance(self.values, str) or np.isscalar(self.values):
            if self.datatype == 'array3d':
                nk = self.dest_model.nlay
            else:
                nk = 1
            self.values = {k: self.values for k in range(nk)}
        elif isinstance(self.values, list):
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
                df = shp2df(f)

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
        df.columns = [c.lower() for c in df.columns]

        # drop any extra unnamed columns from accidental saving of the index on to_csv
        drop_columns = [c for c in df.columns if 'unnamed' in c]
        df.drop(drop_columns, axis=1, inplace=True)
        return df.reset_index(drop=True)


class TransientTabularSourceData(SourceData):
    """Subclass for handling tabular source data that
    represents a time series."""

    def __init__(self, filenames, data_column, datetime_column, id_column,
                 x_col='x', y_col='y', period_stats='mean',
                 length_units='unknown', time_units='unknown', volume_units=None,
                 column_mappings=None,
                 dest_model=None):
        SourceData.__init__(self, filenames=filenames,
                            length_units=length_units, time_units=time_units,
                            volume_units=volume_units,
                            dest_model=dest_model)

        self.data_column = data_column
        self.datetime_column = datetime_column
        self.id_column = id_column
        self.column_mappings = column_mappings
        self.period_stats = period_stats
        self.time_col = datetime_column
        self.x_col = x_col
        self.y_col = y_col

    def get_data(self):

        # aggregate the data from multiple files
        dfs = []
        for i, f in self.filenames.items():
            if f.endswith('.shp') or f.endswith('.dbf'):
                df = shp2df(f)

            elif f.endswith('.csv'):
                df = pd.read_csv(f)

            dfs.append(df)
        df = pd.concat(dfs)
        df.index = pd.to_datetime(df[self.datetime_column])
        # rename any columns specified in config file to required names
        if self.column_mappings is not None:
            df.rename(columns=self.column_mappings, inplace=True)
        df.columns = [c.lower() for c in df.columns]

        # cull data to model bounds
        if 'geometry' not in df.columns:
            df['geometry'] = [Point(x, y) for x, y in zip(df[self.x_col], df[self.y_col])]
        within = [g.within(self.dest_model.bbox) for g in df.geometry]
        df = df.loc[within]

        # sample values to model stress periods
        starttimes = self.dest_model.perioddata['start_datetime'].copy()
        endtimes = self.dest_model.perioddata['end_datetime'].copy()

        # if period ends are specified as the same as the next starttime
        # need to subtract a day, otherwise
        # pandas will include the first day of the next period in slices
        endtimes_equal_startimes = np.all(endtimes[:-1].values == starttimes[1:].values)
        if endtimes_equal_startimes:
            endtimes -= pd.Timedelta(1, unit='d')

        period_data = []
        current_stat = None
        for kper, (start, end) in enumerate(zip(starttimes, endtimes)):
            # missing (period) keys default to 'mean';
            # 'none' to explicitly skip the stress period
            period_stat = self.period_stats.get(kper, current_stat)
            if period_stat is not None and period_stat.lower() == 'none':
                continue
            aggregated = aggregate_dataframe_to_stress_period(df,
                                                              start_datetime=start,
                                                              end_datetime=end,
                                                              period_stat=period_stat,
                                                              id_column=self.id_column,
                                                              data_column=self.data_column
                                                              )
            aggregated['per'] = kper
            period_data.append(aggregated)
        dfm = pd.concat(period_data)

        if self.data_column is not None:
            dfm[self.data_column] *= self.unit_conversion
        dfm.sort_values(by=['per', self.id_column], inplace=True)

        # drop any extra unnamed columns from accidental saving of the index on to_csv
        drop_columns = [c for c in dfm.columns if 'unnamed' in c]
        dfm.drop(drop_columns, axis=1, inplace=True)

        # map x, y locations to modelgrid
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
    if cfg is None and model.cfg['parent'].get('default_source_data'):
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
                if source_variable == 'botm':
                    nlay, nrow, ncol = source_model.dis.botm.array.shape
                    source_array = np.zeros((nlay+1, nrow, ncol))
                    source_array[0] = source_model.dis.top.array
                    source_array[1:] = source_model.dis.botm.array
                    if from_source_model_layers is not None:
                        from_source_model_layers = {k: v+1 for k, v in from_source_model_layers.items()}
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
    if var == 'botm':
        bathy = model.lake_bathymetry
        top = model.load_array(model.cfg[external_files_key]['top'][0])
        lake_botm_elevations = top[bathy != 0] - bathy[bathy != 0]

        # fill missing layers if any
        if len(data) < model.nlay:
            all_surfaces = np.zeros((model.nlay + 1, model.nrow, model.ncol), dtype=float) * np.nan
            all_surfaces[0] = top
            for k, botm in data.items():
                all_surfaces[k + 1] = botm
            all_surfaces = fill_empty_layers(all_surfaces)
            botm = all_surfaces[1:]
        else:
            botm = np.stack([data[i] for i in range(len(data))])

        # adjust layer botms to lake bathymetry (if any)
        # set layer bottom at lake cells to the botm of the lake in that layer
        for k, kbotm in enumerate(botm):
            inlayer = lake_botm_elevations > kbotm[bathy != 0]
            if not np.any(inlayer):
                continue
            botm[k][bathy != 0][inlayer] = lake_botm_elevations[inlayer]

        # fix any layering conflicts and save out botm files
        #if model.version == 'mf6' and model._drop_thin_cells:
        min_thickness = model.cfg['dis'].get('minimum_layer_thickness', 1)
        botm = fix_model_layer_conflicts(top, botm,
                                         minimum_thickness=min_thickness)
        isvalid = verify_minimum_layer_thickness(top, botm,
                                                 np.ones(botm.shape, dtype=int),
                                                 min_thickness)
        if not isvalid:
            raise Exception('Model layers less than {} {} thickness'.format(min_thickness,
                                                                            model.length_units))

        # fill nan values adjacent to active cells to avoid cell thickness errors
        top, botm = fill_cells_vertically(top, botm)
        data = {i: arr for i, arr in enumerate(botm)}
    elif var in ['rech', 'recharge']:
        for per in range(model.nper):
            if per == 0 and per not in data:
                raise KeyError("No recharge input specified for first stress period.")
            if per in data:
                # assign high-k lake recharge for stress period
                # only assign if precip and open water evaporation data were read
                # (otherwise keep original values in recharge array)
                last_data_array = data[per].copy()
                if model.lake_recharge is not None:
                    data[per][model.isbc[0] == 2] = model.lake_recharge[per]
                # zero-values to lak package lakes
                data[per][model.isbc[0] == 1] = 0.
            else:
                if model.lake_recharge is not None:
                    # start with the last period with recharge data; update the high-k lake recharge
                    last_data_array[model.isbc[0] == 2] = model.lake_recharge[per]
                last_data_array[model.isbc[0] == 1] = 0.
                # assign to current per
                data[per] = last_data_array

    elif var in ['ibound', 'idomain']:
        pass
        #for i, arr in data.items():
        #    data[i][model.isbc[i] == 1] = 0.
    elif var in ['hk', 'k']:
        for i, arr in data.items():
            data[i][model.isbc[i] == 2] = model.cfg['model'].get('hiKlakes_value', 1e4)
    elif var in ['ss', 'sy']:
        for i, arr in data.items():
            data[i][model.isbc[i] == 2] = 1.

    # intermediate data
    # set paths to intermediate files and external files
    filepaths = model.setup_external_filepaths(package, var,
                                               model.cfg[package]['{}_filename_fmt'.format(var)],
                                               nfiles=len(data))

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

    # write the top array again, because top was filled
    # with botm array above
    if var == 'botm':
        top_filepath = model.setup_external_filepaths(package, 'top',
                                                      model.cfg[package]['top_filename_fmt'],
                                                      nfiles=1)[0]
        save_array(top_filepath, top,
                   nodata=write_nodata,
                   fmt=write_fmt)
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




