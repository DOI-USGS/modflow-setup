===============
Release History
===============


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
