
Modflow-setup
-----------------------------------------------
Modflow-setup is a Python package for automating the setup of MODFLOW groundwater models from grid-independent source data including shapefiles, rasters, and other MODFLOW models that are geo-located. Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. An external array-based [Flopy](https://github.com/modflowpy/flopy) model instance with the desired packages is created from the sampled source data and configuration settings. MODFLOW input can then be written from the flopy model instance.


### Version 0.6
![Tests](https://github.com/doi-usgs/modflow-setup/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/doi-usgs/modflow-setup/branch/develop/graph/badge.svg?token=aWN47DYeIv)](https://codecov.io/gh/doi-usgs/modflow-setup)
[![PyPI version](https://badge.fury.io/py/modflow-setup.svg)](https://badge.fury.io/py/modflow-setup)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/modflow-setup.svg)](https://anaconda.org/conda-forge/modflow-setup)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/doi-usgs/modflow-setup/develop?urlpath=lab/tree/examples)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)





Getting Started
-----------------------------------------------
For more details, see the [modflow-setup documentation](https://doi-usgs.github.io/modflow-setup/)

Using a [yaml](https://en.wikipedia.org/wiki/YAML)-aware text editor, create a [configuration file](https://doi-usgs.github.io/modflow-setup/latest/config-file.html) similar to one of the examples in the [Configuration File Gallery](https://doi-usgs.github.io/modflow-setup/latest/config-file-gallery.html).

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
See the [Installation Instructions](https://doi-usgs.github.io/modflow-setup/latest/installation.html)


How to cite
-----------------------------------------------
###### Citation for Modflow-setup
Leaf AT and Fienen MN (2022) Modflow-setup: Robust automation of groundwater model construction. Front. Earth Sci. 10:903965. https://doi.org/10.3389/feart.2022.903965

###### Software/Code Citation for Modflow-setup
Leaf, A.T. and Fienen, M.N. (2022). Modflow-setup version 0.1, U.S. Geological Survey Software Release, 30 Sep. 2022. https://doi.org/10.5066/P9O3QWQ1

Applications of Modflow-setup
-----------------------------------------------
Fienen, M.N., Corson-Dosch, N.T., White, J.T., Leaf, A.T. and Hunt, R.J. (2022), Risk-Based Wellhead Protection Decision Support: A Repeatable Workflow Approach. Groundwater, 60: 71-86. https://doi.org/10.1111/gwat.13129

Fienen, M.N., Haserodt, M.J., Leaf, A.T., and Westenbroek, S.M., 2022, Simulation of regional groundwater flow and groundwater/lake interactions in the Central Sands, Wisconsin: U.S. Geological Survey Scientific Investigations Report 2022–5046, 111 p., https://doi.org/10.3133/sir20225046.

Leaf, A.T., Duncan, L.L., Haugh, C.J., Hunt, R.J., and Rigby, J.R., 2023, Simulating groundwater flow in the Mississippi Alluvial Plain with a focus on the Mississippi Delta: U.S. Geological Survey Scientific Investigations Report 2023–5100, 143 p., https://doi.org/10.3133/sir20235100.

Workflow examples
-----------------------------------------------
Fienen, M.N, and Corson-Dosch, N.T., 2021, Groundwater Model Archive and Workflow for Neversink/Rondout Basin, New York, Source Water Delineation: U.S. Geological Survey data release, https://doi.org/10.5066/P9HWSOHP.

Leaf, A.T., Duncan, L.L., and Haugh, C.J., 2023, MODFLOW 6 models for simulating groundwater flow in the Mississippi Embayment with a focus on the Mississippi Delta: U.S. Geological Survey data release, https://doi.org/10.5066/P971LPOB.

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
resulting from the authorized or unauthorized use of the software. It is the responsibility of the user to check the accuracy of the results.

Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government.
