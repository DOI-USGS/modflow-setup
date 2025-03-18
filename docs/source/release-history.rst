===============
Release History
===============

Version 0.6.1 (2025-03-17)
----------------------------------------
* fix 'TemporalReference' issue related to change in Flopy code
* fix issue with Mover Package SFR connections across LGR model interfaces (mover.py::get_sfr_package_connections)); follow-up fix to fbe1bb1 to handle cases where parent model SFR reach along the LGR interface has both upstream and downstream connections to the inset model.

Version 0.6.0 (2025-01-06)
----------------------------------------

* Add support for local grid refinement (LGR) in a subset of the parent model layers.
    * Replace parent_start/end layer configuration input with "vertical_refinement" in each parent layer (int, list or dict), which gets translated to ncppl input to the Flopy Lgr utility.
    * Child model bottom and parent model top are exactly aligned with no overlap or gaps in numerical grid.
    * On setup of the LGR grid, the parent model cell tops/bottoms are collapsed to zero thickness within the child model area(s).
    * Recharge, SFR and non-Well basic stress boundary conditions are only applied to the Child model (assuming that these represent surface or near-surface features that should only be represented in the child model).
    * Wells with > 50% of their open interval intersecting the child model domain get assigned to the child model.
    * Wells with > 50% of their open interval intersecting the parent model get assigned to the parent model.

Version 0.5.0 (2024-03-08)
----------------------------------------
* Improvements to rotated grid generation
  * add support for generating grids with a specified rotation around features of interest
  * add support for rotated LGR models (LGR parent and inset with same rotation)
  * add additional validation checks in grid setup
  * fix issue related to SFR layer assignment in cases with multiple inactive layers below the lowest active cell
  * fix issue with reading transient array-based source data from multiple rasters
  * other miscellaneous fixes

Version 0.4.0 (2024-01-15)
----------------------------------------
* Improvements to lake package

  * Add automatic writing of lake polygon and lake cell connection shapefiles
  * Add name_column arg to shapefile input (which adds names to lake auxiliary tables)
  * Allow PRISM input to be specified for all lakes (via single filename instead of dictionary)
  * Allow specification of lakes_shapefile: without include_ids: item
  * Move output tables to tables/ folder
* Pre-defined stress periods

  * Allow stress period data to be pre-defined in a CSV file, which allows for more complicated or irregular stress period configurations that would otherwise require many group blocks; for example 7-day periods that always start on the same day of the year, which results in an extra period of 1 or 2 days at the end of each year.
* Bug fixes

  * fix issue with identifier column dtypes, so that NHDPlus Hi-Res COMIDs (which require 64-bits as integers) work more robustly on Windows.
  * in case of pre-defined (csvfile) stress periods, base perlen on end_datetime - start_datetime (what you see is what you get, and so that gaps between stress periods don't affect perlen); add trap for missing required columns
  * see commit history for other misc. fixes

Version 0.3.1 (2023-08-17)
----------------------------------------
* change 'boundname_col' argument in basic stress CSV input to 'boundname_column' for consistency with other inputs.
* fix issue with layer assignment in SFR Package setup where idomain/ibound array wasn't getting passed to the SFRmaker `assign_layers()` function, which can be problematic for models with extensive inactive cells in their upper layers.
* fix issue with model grid setup where `.prj` file for bounding box shapefile wasn't being written.
* bug fixes for compatibility with `flopy>=3.4``
* fixes to `grid.rasterize()` to better handle 64-bit integer and `object` dtypes
* update example configuration files to use new length unit arguments in `sfrmaker>=0.11.1`

Version 0.3.0 (2023-07-25)
----------------------------------------

* Transient input to basic stress packages can now be supplied via a ``csvfile:`` block. Transient
  values for the package variables are associated with the shapefile feature extents via an
  ``id_column`` of values that match values in the shapefile id_column. Input can be mixed between
  transient ``csvfile:`` input and static input supplied via rasters (that varies spatially)
  or uniform global values. See the Basic stress packages section in the documentation for more details.
* add automatic reprojection of NetCDF source data files
* most Soil Water Balance code models should be work as-is
* added ``crs:`` configuration file item to specify the coordinate reference system for NetCDF files
  where coordinate reference information can't be parsed.
* add support for inner and outer CSV file output in MODFLOW 6 IMS options (remap input to work with Flopy)
* refactor ``grid.rasterize()`` function to work with standard Flopy ``StructuredGrid`` (s)
* refactor MODFLOW 6 head observations setup
* add ``allow_obs_in_bc_cells`` option to allow observations in boundary condition cells (previously not allowed).
* put modflow-setup specific options in an ``mfsetup_options:`` sub-block, consistent with other packages
* fixes to adapted to breaking changes with pandas 2.0 and revised flopy modelgrid/crs interface
* fix ``AttributeError`` issue with model name not getting passed to flopy
* some fixes to model version reporting in MODFLOW input file headers

Version 0.2.0 (2023-02-06)
----------------------------------------
* add minimal support for MODFLOW-2000 parent models and variably-spaced structured grids
    * relax requirement that inset cells align with parent cells
    * add package translations between mf6 and mf2k
* remove all basic stress package bcs from inactive cells prior to write
* fix to allow for no epsg input argument to setup_grid (crs is now favored)
* add support for virtual raster (`*.vrt` file) input
* add support for parent MODFLOW model Name files with blank lines
* other fixes to adapt to breaking changes in numpy 1.24, pandas, inspect, collections and SFRmaker

Version 0.1.0 Initial release (2022-09-30)
-----------------------------------------------
* support for constructing MODFLOW-NWT or MODFLOW-6 models from scratch
* supported source dataset formats include GeoTiff, Arc-Ascii grids, shapefiles, NetCDF, and CSV files
* automatic reprojection of source datasets that have CRS information (GeoTiffs, shapefiles, etc.)
* supported MODFLOW-NWT packages: DIS, BAS6, OC, UPW, RCH, GHB, SFR2, LAK, WEL, MNW2, HYD, GAGE, NWT, CHD, DRN, GHB, RIV
* supported MODFLOW-6 packages: DIS, IC, OC, NPF, RCHA, SFR, LAK, WEL, OBS, IMS, TDIS, CHD, DRN, GHB, RIV
* Lake observations are set up automatically (one output file per lake); basic stress package observations are also setup up automatically
* SFR observations can be set up via an ``observations`` block in the SFR package ``source_data``
* Telescopic mesh refinement (TMR) insets from MODFLOW-NWT or MODFLOW 6 models
    * support for specified head or flux perimeter boundaries from the parent model solution
    * perimeter boundaries can be along the model bounding box perimeter, or an irregular polygon of the active model area
    * model input can also be supplied from grid-independent source data, parent model package input (except for SFR, LAK, CHD, DRN, GHB, RIV), or MODFLOW binary output (perimeter boundaries and starting heads)

* Dynamically coupled local grid refinement (LGR) insets supported for MODFLOW-6 models.
    * The water mover (MVR) package is set up automatically at the simulation level for LGR models with SFR packages in the parent and inset model.

* see the `Configuration File Gallery`_ or the ``*.yml`` configuation files in mfsetup/tests/data folder for examples of valid input to modflow-setup

.. _Configuration File Gallery: https://doi-usgs.github.io/modflow-setup/docs/build/html/examples.html#configuration-file-gallery
