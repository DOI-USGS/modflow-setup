===============
Release History
===============

Version 0.1 Initial release (2022-xx-xx)
----------------------------------------
* support for constructing MODFLOW-NWT or MODFLOW-6 models from scratch
* supported source dataset formats include GeoTiff, Arc-Ascii grids, shapefiles, NetCDF, and CSV files
* automatic reprojection of source datasets that have CRS information (GeoTiffs, shapefiles, etc.)
* supported MODFLOW-NWT packages: DIS, BAS6, OC, UPW, RCH, GHB, SFR2, LAK, WEL, MNW2, HYD, GAGE, NWT, CHD, DRN, GHB, RIV
* supported MODFLOW-6 packages: DIS, IC, OC, NPF, RCHA, SFR, LAK, WEL, OBS, IMS, TDIS, CHD, DRN, GHB, RIV
* Lake observations are set up automatically (one output file per lake); basic stress package observations are also setup up automatically
* SFR observations can be set up via ``observations`` block in the SFR package ``source_data``
* Telescopic mesh refinement (TMR) insets from MODFLOW-NWT or MODFLOW 6 models
    * support for specified head or flux perimeter boundaries from the parent model solution
    * perimeter boundaries can be along the model bounding box perimeter, or an irregular polygon of the active model area
    * model input can also be supplied from grid-independent source data, parent model package input (except for SFR, LAK, CHD, DRN, GHB, RIV), or MODFLOW binary output (perimeter boundaries and starting heads)

* Local grid refinement (LGR) insets supported for MODFLOW-6 models.
    * The water mover (MVR) package is set up automatically at the simulation level for LGR models with SFR packages in the parent and inset model.

* see the `Configuration File Gallery`_ or the ``*.yml`` configuation files in mfsetup/tests/data folder for examples of valid input to modflow-setup

.. _Configuration File Gallery: https://doi-usgs.github.io/modflow-setup/docs/build/html/examples.html#configuration-file-gallery
