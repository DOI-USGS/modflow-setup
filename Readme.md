
modflow-setup
-----------------------------------------------
Package to facilitate automated setup of MODFLOW models, from source data including shapefiles, rasters, and other MODFLOW models that are geo-located. Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. An external array-based [flopy](https://github.com/modflowpy/flopy) model instance with the desired packages is created from the sampled source data and default settings. MODFLOW input can then be written from the flopy model instance.


### Version 0.1
![Tests](https://github.com/aleaf/modflow-setup/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/aleaf/modflow-setup/branch/master/graph/badge.svg)](https://codecov.io/gh/aleaf/modflow-setup)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aleaf/modflow-setup/develop?urlpath=lab/tree/examples)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)





Getting Started
-----------------------------------------------
For more details, see the [modflow-setup documentation](https://aleaf.github.io/modflow-setup/)

Using a [yaml](https://en.wikipedia.org/wiki/YAML)-aware text editor, create a [configuration file](https://aleaf.github.io/modflow-setup/latest/config-file.html) similar to one of the examples in the [Configuration File Gallery](https://aleaf.github.io/modflow-setup/latest/config-file-gallery.html).

The yaml file summarizes source data and parameter settings for setting up the various MODFLOW packages. To set up the model:

```
from mfsetup import MFnwtModel, MF6model

m = MF6model.setup_from_yaml(<path to configuration file>)
```
where `m` is a [flopy](https://github.com/modflowpy/flopy) MODFLOW-6 model instance that is returned. The MODFLOW input files can be written from the model instance:

```
m.simulation.write_simulation()
```

MODFLOW-NWT version:

```
m = MFnwtModel.setup_from_yaml(<path to configuration file>)
m.write_input()
```

Installation
-----------------------------------------------
See the [Installation Instructions](https://aleaf.github.io/modflow-setup/latest/installation.html)

MODFLOW Resources
-----------------------------------------------

+ [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model)
+ [Online Guide to MODFLOW-NWT](https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.
