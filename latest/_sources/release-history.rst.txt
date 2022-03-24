===============
Release History
===============

Version 0.1 Initial release (2022-04-01)
----------------------------------------
* support for constructing MODFLOW-NWT or MODFLOW-6 models from scratch
* supported source dataset formats include GeoTiff, Arc-Ascii grids, shapefiles, NetCDF, and CSV files
* automatic reprojection of source datasets that have CRS information (GeoTiffs, shapefiles, etc.)
* model input can also be supplied from parent model package input or MODFLOW binary output
* supported MODFLOW-NWT packages: DIS, BAS6, OC, UPW, RCH, GHB, SFR2, LAK, WEL, MNW2, HYD, GAGE, NWT
* supported MODFLOW-6 packages: DIS, IC, OC, NPF, RCHA, SFR, LAK, WEL, OBS, IMS, TDIS, CHD, DRN, GHB, RIV
* Lake observations are set up automatically (one output file per lake)
* SFR observations can be set up via ``observations`` block in the SFR package ``source_data``
* Telescopic mesh refinement (TMR) insets from MODFLOW-NWT or MODFLOW 6 models
    * support for specified head boundaries from parent head solution
    * specified head boundaries can be along model perimeter or irregular polygon of model active area
* Local grid refinement (LGR) insets supported for MODFLOW-6 models.
  * The water mover (MVR) package is set up automatically at the simulation level for LGR models with SFR packages in the parent and inset model.
* see .yml configuation files in mfsetup/tests/data folder for examples of valid input to modflow-setup
