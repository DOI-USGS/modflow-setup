
modflow-setup
-----------------------------------------------
Package to facilitate automated setup of MODFLOW models, from source data including shapefiles, rasters, and other MODFLOW models that are geo-located. Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. An external array-based [flopy](https://github.com/modflowpy/flopy) model instance with the desired packages is created from the sampled source data and default settings. MODFLOW input can then be written from the flopy model instance.


### Version 0.1
[![Build Status](https://travis-ci.org/aleaf/modflow-setup.svg?branch=master)](https://travis-ci.org/aleaf/modflow-setup)
[![Build status](https://ci.appveyor.com/api/projects/status/5l11v18na9p28olh/branch/master?svg=true)](https://ci.appveyor.com/project/aleaf/modflow-setup/branch/master)
[![codecov](https://codecov.io/gh/aleaf/modflow-setup/branch/master/graph/badge.svg)](https://codecov.io/gh/aleaf/modflow-setup)





Getting Started
-----------------------------------------------
For more details, see the [modflow-setup documentation](https://aleaf.github.io/modflow-setup/)

Using a [yaml](https://en.wikipedia.org/wiki/YAML)-aware text editor, create a configuration file similar to the included test files:

* [MODFLOW-NWT example](https://github.com/aleaf/modflow-setup/blob/master/mfsetup/tests/data/mfnwt_inset_test.yml)
* [MODFLOW-6 example](https://github.com/aleaf/modflow-setup/blob/master/mfsetup/tests/data/shellmound.yml)

The yaml file summarize source data and parameter settings for setting up the various MODFLOW packages. To set up the model:

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


### Bugs

If you think you have discovered a bug in modflow-setup in which you feel that the program does not work as intended, then we ask you to submit a [Github issue](https://github.com/aleaf/modflow-setup/labels/bug).


Installation
-----------------------------------------------

**Python versions:**

modflow-setup requires **Python** 3.6 (or higher)

**Dependencies:**  
pyaml  
numpy   
scipy  
xarray  
pandas  
fiona  
rasterio  
rasterstats  
shapely  
rtree  
pyproj  
flopy   
sfrmaker

### Install python and dependency packages
Download and install the [Anaconda python distribution](https://www.anaconda.com/distribution/).
Open an Anaconda Command Prompt on Windows or a terminal window on OSX.
From the root folder for the package (that contains `requirements.yml`), install the above packages from `requirements.yml`.

```
conda env create -f requirements.yml
```
activate the environment:

```
conda activate mfsetup
```

### Install to site_packages folder
```
python setup.py install
```
### Install in current location (to current python path)
(i.e., for development)  

```  
pip install -e .
```



MODFLOW Resources
-----------------------------------------------

+ [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model)



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

