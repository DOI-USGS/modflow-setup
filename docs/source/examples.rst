Examples
========

Configuration File Gallery
--------------------------
Below are example (valid) configuration files from the modflow-setup test suite. The yaml files and the datasets they reference can be found under ``modflow-setup/mfsetup/tests/data/``.

Shellmound test case
^^^^^^^^^^^^^^^^^^^^
* 13 layer MODFLOW-6 model with no parent model
* 9 layers specified with raster surfaces; with remaining 4 layers subdividing the raster surfaces
* `vertical pass-through cells`_ at locations of layer pinch-outs (``drop_thin_cells: True`` option)
* variable time discretization
* model grid aligned with the `National Hydrologic Grid`_
* recharge read from NetCDF source data
* SFR network created from custom hydrography
* WEL package created from CSV input


.. literalinclude:: ../../mfsetup/tests/data/shellmound.yml
    :language: yaml
    :linenos:

Pleasant Lake test case
^^^^^^^^^^^^^^^^^^^^^^^
* MODFLOW-6 model with local grid refinement (LGR)
* LGR parent model is itself a Telescopic Mesh Refinment (TMR) inset from a MODFLOW-NWT model
* Layer 1 in TMR parent model is subdivided evenly into two layers in LGR model (``botm: from_parent: 0: -0.5``). Other layers mapped explicitly between TMR parent and LGR model.
* starting heads from LGR parent model resampled from binary output from the TMR parent
* rch, npf, sto, and wel input copied from parent model
* SFR package constructed from an NHDPlus v2 dataset (path to NHDPlus files in the same structure as the `downloads from the NHDPlus website`_)
* head observations from csv files with different column names
* LGR inset extent based on a buffer distance around a feature of interest
* LGR inset dis, ic, npf, sto and rch packages copied from LGR parent
* WEL package created from custom format
* Lake package created from polygon features, bathymetry raster, stage-area-volume file and climate data from `PRISM`_.
* Lake package observations set up automatically (output file for each lake)

LGR parent model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../mfsetup/tests/data/pleasant_lgr_parent.yml
    :language: yaml
    :linenos:

pleasant_lgr_inset.yml
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../mfsetup/tests/data/pleasant_lgr_inset.yml
    :language: yaml
    :linenos:

Pleasant Lake MODFLOW-NWT test case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* MODFLOW-NWT TMR inset from a MODFLOW-NWT model
* Layer 1 in parent model is subdivided evenly into two layers in the inset model (``botm: from_parent: 0: -0.5``). Other layers mapped explicitly between TMR parent and LGR model.
* starting heads resampled from binary output from the TMR parent
* rch, npf, sto, and wel input copied from parent model
* SFR package constructed from an NHDPlus v2 dataset (path to NHDPlus files in the same structure as the `downloads from the NHDPlus website`_)
* HYDMOD package for head observations from csv files with different column names
* LGR inset extent based on a buffer distance around a feature of interest
* LGR inset dis, ic, npf, sto and rch packages copied from LGR parent
* WEL package created from custom format
* Lake package created from polygon features, bathymetry raster, stage-area-volume file and climate data from `PRISM`_.
* Lake package observations set up automatically (output file for each lake)
* GHB package created from polygon feature and DEM raster

.. literalinclude:: ../../mfsetup/tests/data/pleasant_nwt_test.yml
    :language: yaml
    :linenos:

.. _downloads from the NHDPlus website: https://nhdplus.com/NHDPlus/NHDPlusV2_data.php
.. _vertical pass-through cells: https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.1.0.pdf
.. _PRISM: http://www.prism.oregonstate.edu/
.. _National Hydrologic Grid: https://www.sciencebase.gov/catalog/item/5a95dd5de4b06990606a805e