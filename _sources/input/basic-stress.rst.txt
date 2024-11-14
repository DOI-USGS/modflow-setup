=======================================================================================
Specifying boundary conditions with the 'basic' MODFLOW stress packages
=======================================================================================

This page describes configuration file input for the basic MODFLOW stress packages, including
the CHD, DRN, GHB, RCH, RIV and WEL packages. The EVT package is not currently supported by Modflow-setup. The supported packages can be broadly placed into two categories. Feature or list-based packages such as CHD, DRN, GHB, RIV and WEL often represent discrete phenomena such as surface water features, pumping wells, or even lines that denote a perimeter boundary. Input to  these packages in MODFLOW is tabular, consisting of a table for each stress period, with rows specifying stresses at individual grid cells representing the boundary features. In contrast, continuous or grid-based packages represent a stress field that applies to a large area, such as areal recharge. In past versions of MODFLOW, input to these packages was array-based, with values specified for all model cells, at each stress period. In MODFLOW 6, input to these packages can be array or list-based. The Recharge (RCH) Package is currently the only grid-based stress package supported by Modflow-setup. In keeping with the current structured grid-based paradigm of Modflow-setup, Modflow 6 recharge input is generated for the array-based recharge package (Langevin and others, 2017).



List-based basic stress packages
-------------------------------------

Input for list-based basic stress packages follows a similar pattern to other packages.

* Package blocks are named using the 3 letter MODFLOW abbrieviation for the Package in lower case (e.g. ``chd:``, ``ghb:``, etc.).
* Sub-blocks within the package block include:

    * ``options:`` for specifying Modflow 6 options, exactly as they are described in the input instructions (Langevin and others, 2017).
    * ``source_data:`` for specifying grid-independent source data to be mapped to the model discretization, in addition to other package input. ``source_data:`` in turn can have the following sub-blocks and items:
        * A ``shapefile:`` block for specifying shapefile input that maps the boundary condition features in space. Items in the shapefile block include

            * ``filename:`` path to the shapefile
            * ``boundname_col:`` column in shapefile with feature names to be applied as `boundnames` in Modflow 6 input
            * ``all_touched:`` argument to :func:`rasterio.features.rasterize` that specifies whether all intersected grid cells should be included, or just the grid cells with centers inside the feature.
            * One or more variable columns: Optionally the shapefile can also supply steady-state variable values by feature in attribute columns named for the variables (e.g. ``'head'``, ``'bhead'``, etc.)

            Example:

            .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                :language: yaml
                :start-at: shapefile:
                :end-before: csvfile:

        * A ``csvfile:`` block for specifying point feature locations or time-varying values for the variables. Items in the shapefile block include

            * ``filename:`` or `filenames:` path(s) to the csv file(s)
            * ``id_column``: unique identifier associated with each feature
            * ``datetime_column:`` date-time associated with each stress value
            * ``end_datetime_column:`` date-time associated with the end of stress value (optional; for rates that extend across more than one model stress period. If this is specified, ``datetime_column:`` is assumed to indicate the date-time associated with the start of each stress value.
            * ``x_col:`` feature x-coordinate (WEL package only; default ``'x'``)
            * ``y_col:`` feature y-coordinate (WEL package only; default ``'y'``)
            * ``length_units:`` length units associated with the stress value (optional; if omitted no conversion is performed)
            * ``time_units:`` time units associated with the stress value (WEL package only; optional; if omitted no conversion is performed)
            * ``volume_units:`` value-units associated with the stress value (e.g. `gallons`) in lieu of length-based volume units (e.g., `cubic feet`) (WEL package only; optional; if omitted volumes are assumed to be in model units of L\ :sup:`3` and no conversion is performed)
            * ``boundname_col:`` column in shapefile with feature names to be applied as `boundnames` in Modflow 6 input
            * one or more columns for the package variables, specified in the format of ``<variable>_col``, where ``<variable>`` is an input variable for the package; for example ``head_col`` for the Constant Head Package, or ``cond_col`` for the Drain or GHB packages.
            * ``period_stats:`` a sub-block that is used to specify mapping of the input data to the model temporal discretization. Items within period stats are numbered by stress period, with the entry for each item specifying the temporal aggregation. Currently, two options are supported:
                * aggregation of measurements falling within a stress period. For example, assigning the mean value of all input data points within the stress period. In this case, the aggregration method is simply specified as a string. While ``mean`` is typical, any of the standard numpy aggregators can be use (``min``, ``max``, etc.)
                * aggregation of measurements from an arbitrary time window. For example, applying a long-term mean to a steady-state stress period, or transient period representing a different time window. In this case three items are specified-- the aggregation method, the start date, and end date (e.g. ``[mean, 2000-01-01, 2017-12-31]``)

            Example:

            .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                :language: yaml
                :start-at: csvfile:
                :end-before: # Drain Package


        * Additional sub-blocks or items for specifying values for each variable
            * In general, these sub-blocks are named for the variable (e.g. ``bhead:``).
            * Scalar values (items) can be specified in model units, and are applied globally to the variable.

                Example:

                .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                    :language: yaml
                    :start-at: cond:
                    :end-at: cond:

            * Rasters can be used to specify steady-state values that vary in space; values supplied with a raster are mapped to the model grid using zonal statistics. If the raster contains projection information (GeoTIFFs are preferred in part because of this), reprojection to the model coorindate reference system (CRS) will be performed automatically as needed. Otherwise, the raster is assumed to be in the model projection. Units can optionally be specified and automatically converted; otherwise, the raster values are assumed to be in the model units. Items in the raster block include:

                * ``filename:`` or `filenames:` path(s) to the raster
                * ``length_units:`` (or ``elevation_units``; optional): length units of the raster values
                * ``time_units:`` (optional): time units of the raster values (``cond`` variable only)
                * ``stat:`` (optional): zonal statistic to use in sampling the raster (defaults are listed for each variable in the :ref:`Configuration defaults`)

                Example:

                .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                    :language: yaml
                    :start-at: stage:
                    :end-before: mfsetup_options:

            * **Not implemented yet:** NetCDF input for gridded values that vary in time and space. Due to the lack of standardization in NetCDF coordinate reference information, automatic reprojection is currently not supported for NetCDF files; the data are assumed to be in the model CRS.
    * ``mfsetup_options:`` Configuration options for Modflow-setup. General options that apply to all basic stress packages include:
            * ``external_files:`` Whether to write the package input as external text arrays or tables (i.e., with ``open/close`` statements). By default ``True`` except in the case of list-based or tabular files for MODFLOW-NWT models, which are not supported. Adding support for this may require changes to Flopy, which handles external list-based files differently for MODFLOW-2005 style models.
            * ``external_filename_fmt:`` Python string format for external file names. By default, ``"<package or variable abbreviation>_{:03d}.dat"``. which results in filenames such as ``wel_000.dat``, ``wel_001.dat``, ``wel_002.dat``... for stress periods 0, 1, and 2, for example.

            Other Modflow-setup options specific to individual packages are described below.

Constant Head (CHD) Package
++++++++++++++++++++++++++++++
Input consists of specified head values that may vary in time or space.

    **Required input**

    * parent model head solution --or--
    * shapefile of features --or--
    * parent model package (not implemented yet)
    * at least steady-state head values through one of the methods below

    **Optional input**

    * raster to specify steady state elevations by cell (for supplied shapefile)
    * shapefile or csv to specify steady elevations by feature
    * csv to specify transient elevation by feature (needs to be referenced to features in shapefile)

    **Examples**
    (also see the :ref:`Configuration File Gallery`)

    Setting up a Constant Head package with perimeter fluxes from a parent model (Note: an additional ``source_data`` block can be added to represent other features inside of the model perimeter, as below):

    .. literalinclude:: ../../../mfsetup/tests/data/pfl_nwt_test.yml
        :language: yaml
        :start-at: chd:

    Setting up a Constant Head package from features specified in a shapefile,
    and time-varing heads specified in a csvfile:

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
        :language: yaml
        :start-after: # Constant Head Package
        :end-before: # Drain Package


Drain DRN Package
++++++++++++++++++
Input consists of elevations and conductances that may vary in time or space.

    **Required input**

    * shapefile of features  --or--
    * parent model package (not implemented yet)
    * at least steady-state head and conductance values through one of the methods below

    **Optional input**

    * global conductance value specified directly
    * raster to specify steady state elevation by cell (for supplied shapefile)
    * shapefile or csv to specify steady elevations by feature
    * csv to specify transient elevation by feature (needs to be referenced to features in shapefile)

    **Examples**
    (also see the :ref:`Configuration File Gallery`)

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
        :language: yaml
        :start-after: # Drain Package
        :end-before: # General Head Boundary Package

General Head Boundary (GHB) Package
+++++++++++++++++++++++++++++++++++++
Input consists of head elevations and conductances that may vary in time or space.

    **Required input**

    * shapefile of features --or--
    * parent model package (not implemented yet)
    * at least steady-state head and conductance values through one of the methods below

    **Optional input**

    * global conductance value specified directly
    * shapefile or csv to specify steady elevations and conductances by feature --or--
    * rasters to specify steady state elevations or conductances by cell (for supplied shapefile)
    * csv to specify transient elevations or conductances by feature (needs to be referenced to features in shapefile)

    **Examples**
    (also see the :ref:`Configuration File Gallery`)

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
        :language: yaml
        :start-after: # General Head Boundary Package
        :end-before: # River Package

River (RIV) package
++++++++++++++++++++
Input consists of stages, river bottom elevations and conductances,
 that may vary in time or space.

    **Required input**

    * shapefile of features --or--
    * ``to_riv:`` block under ``sfrmaker_options:`` with an ``sfr:`` block (see configuration gallery)
    * parent model package (not implemented yet)

    **Optional input**

    * global conductance value specified directly
    * ``default_rbot_thick`` argument to set a uniform riverbed thickness (``rbot = stage - uniform thickness``)
    * shapefile or csv to specify steady heads, conductances and rbots by feature --or--
    * rasters to specify steady heads, conductances and rbots by cell (for supplied shapefile)
    * csv to specify transient heads, conductances and rbots by feature (needs to be referenced to features in shapefile)

    **Examples**
    (also see the :ref:`Configuration File Gallery`)

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
        :language: yaml
        :start-after: # River Package
        :end-before: # Well Package

    Example of setting up the RIV package using SFRmaker (via the ``sfr:`` block):

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound_tmr_inset.yml
        :language: yaml
        :start-at: sfr:
        :end-at: to_riv:


Well (WEL) Package
++++++++++++++++++++
Input consists of flux rates that may vary in time or space.

    **Required input**

    * parent model cell by cell flow solution (not implemented yet) --or--
    * parent model WEL package
    * steady-state or transient flux values through one of the methods below

    **Optional input**

    * temporal discretization (default is to use the average rate(s) for each stress period)
    * vertical discretization (default is to distribute fluxes vertically by the individual transmissivities of the intersection(s) of the well open interval with the model layers.)

    **Flux input options with examples**
    (also see the :ref:`Configuration File Gallery`)

    * Fluxes translated from a parent model WEL package
        * this input option is very simple. A parent model with a well package is needed, and ``default_source_data: True`` must be specified in the ``parent:`` block. Then, fluxes from the parent model are simply mapped to the inset model grid, based on the parent model cell centers, and the stress period mappings specified in the ``parent:`` block. Well package options can still be specified in a ``wel:`` block.
        * Examples:

            .. literalinclude:: ../../../mfsetup/tests/data/pleasant_mf6_test.yml
                :language: yaml
                :lines: 119-123

    * CSV input from one or more files (``csvfiles:`` block)
        * multiple files can be specified using a list, but column names and units must be consistent
        * input for column names and units is the same for the general ``csvfile:`` block described above
        * temporal discretization is specified using a ``period_stats:`` sub-block
        * spatial discretization for open intervals spanning multiple layers is specified using a ``vertical_flux_distribution:`` sub-block
        * Examples:

            .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                :language: yaml
                :start-after: # Well Package
                :end-before: # Output Control Package

    * Perimeter boundary fluxes from a parent model solution:

            .. literalinclude:: ../../../mfsetup/tests/data/shellmound_tmr_inset.yml
                :language: yaml
                :start-at: wel:


        Similar to the Constant Head Package, a ``perimeter_boundary`` block can be mixed with the other input blocks described here to simulate pumping or injection inside of the model perimeter.

    * ``wdnr_dataset`` block
        .. note::
            This is a custom option from early versions of Modflow-setup, and is likely to be generalized into a combined shapefile (or CSV site information file) and CSV timeseries input option similar to the other basic stress packages.

        * site information is specified in a shapefile formatted like ``csls_sources_wu_pts.shp`` below
        * pumping rates are specified by month in a CSV file formatted like ``master_wu.csv`` below
        * temporal discretization is specified with a ``period_stats:`` block similar to the ``csvfiles:`` option
        * vertical discretization is specified with a ``vertical_flux_distribution:`` block similar to the ``csvfiles:`` option

        * Example:

            .. literalinclude:: ../../../mfsetup/tests/data/pfl_nwt_test.yml
                :language: yaml
                :lines: 113-118

    **The** ``vertical_flux_distribution:`` **sub-block**
        * This sub-block specifies how Well Packages fluxes should be distributed vertically.
        * Items/options include:
            * ``across_layers:`` If ``True``, fluxes for a well will be put in the layer containing the open interval midpoint. If ``False``, fluxes will be distributed to the layers intersecting the well open interval.
            * ``distribute_by:`` ``'transmissivity'`` (default) to distribute fluxes based on the transmissivities of open interval/layer intersections; ``'thickness'`` to distribute fluxes based on intersection thicknesses. Only relevant with ``across_layers: True``.
            * ``minimum_layer_thickness:`` Minimum layer thickness for placing a well (by default 2 model length units). Wells in layers thinner than this will be relocated to the thickess layers at their row, column locations. If no thicker layers exist at the row, column location, the wells are dropped, and reported in *<model name>_dropped_wells.csv*.


Grid-based basic stress packages
-------------------------------------
The Recharge (RCH) Package is currently the only grid-based stress package supported by Modflow-setup.


Recharge (RCH) Package
++++++++++++++++++++++++

Direct input
@@@@@@@@@@@@@@@@
As with other grid-based input such as aquifer properties, input to the recharge package can be specified directly as it would in Flopy. This may be useful for setting up a test model quickly. For example, a single scalar value could be entered to apply to all locations across all periods:

.. code-block:: yaml

    rch:
      recharge: 0.001

Or global scalar values could be entered by stress period:

.. code-block:: yaml

    rch:
      recharge:
        0: 0.001
        1: 0.01

In the above example, ``0.01`` would be also be applied to all subsequent stress periods.

Grid-independent input
@@@@@@@@@@@@@@@@@@@@@@@@@@@
Modflow-setup currently supports three methods for entering spatially-referenced recharge input not mapped to the model grid.

    * Recharge translated from a parent model RCH package
        * This input option is very simple. A parent model with a recharge package is needed, and ``default_source_data: True`` must be specified in the ``parent:`` block. Then, fluxes from the parent model are simply mapped to the inset model grid, based on the parent model cell centers, and the stress period mappings specified in the ``parent:`` block. Recharge package options can still be specified in a ``rch:`` block.

    * Raster input by stress period
        * A raster of spatially varying recharge values can be supplied for one or more model stress periods. Similar to the direct input, specified recharge will be applied to subsequent periods were recharge is not specified.
        * If the raster contains projection information (GeoTIFFs are preferred in part because of this), any reprojection to the model coorindate reference system (CRS) will be performed automatically as needed. Otherwise, the raster is assumed to be in the model projection.
        * Input items include:
            * ``length_units:`` input recharge length units (optional; if omitted no conversion is performed)
            * ``time_units:`` input recharge time units (optional; if omitted no conversion is performed)
            * ``mult:`` option multiplier value that applies to all stress periods.
            * ``resample_method:`` method for resampling the data from the source grid to model grid. (optional; by default, ``'nearest'``)

        * Examples:

            .. literalinclude:: ../../../mfsetup/tests/data/pfl_nwt_test.yml
                :language: yaml
                :lines: 99-106

    * NetCDF input
        * NetCDF input can be supplied for gridded values that vary in time and space.
        * Automatic reprojection is supported for Climate Forecast (CF) 1.8-compliant netcdf files (that work with the :py:meth:`pyproj.CRS.from_cf() <pyproj.crs.CRS.from_cf>` constructor), or files that have a `'crs_wkt'` or `'proj4_string'` grid mapping variable (the latter includes many or most Soil Water Balance Code models).
        * Otherwise, coordinate reference information can be supplied via the ``crs:`` item (using any valid input to :py:class:`pyproj.crs.CRS`), and the data will be reprojected to the model coordinate reference system.

        * Input items include:
            * ``variable:`` name of variable in NetCDF file containing the recharge values.
            * ``length_units:`` input recharge length units (optional; if omitted no conversion is performed)
            * ``time_units:`` input recharge time units (optional; if omitted no conversion is performed)
            * ``crs``: coordinate reference system (CRS) of the netcdf file (optional; only needed if the NetCDF file is in a different CRS than the model *and* automatic reprojection from the internal `grid mapping <http://cfconventions.org/cf-conventions/cf-conventions.html#grid-mappings-and-projections>`_ isn't working.
            * ``resample_method:`` method for resampling the data from the source grid to model grid. (optional; by default, ``'nearest'``)
            * ``period_stats:`` a sub-block that is used to specify mapping of the input data to the model temporal discretization. Items within period stats are numbered by stress period, with the entry for each item specifying the temporal aggregation. Currently, two options are supported:
                * aggregation of measurements falling within a stress period. For example, assigning the mean value of all input data points within the stress period. In this case, the aggregration method is simply specified as a string. While ``mean`` is typical, any of the standard numpy aggregators can be use (``min``, ``max``, etc.)
                * aggregation of measurements from an arbitrary time window. For example, applying a long-term mean to a steady-state stress period, or transient period representing a different time window. In this case three items are specified-- the aggregation method, the start date, and end date (e.g. ``[mean, 2000-01-01, 2017-12-31]``; see below for an example)

        * Examples:

            .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
                :language: yaml
                :start-after: # Recharge Package
                :end-before: # Streamflow Routing Package
