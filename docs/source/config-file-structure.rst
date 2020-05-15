Configuration file structure
----------------------------
In general, the configuration file structure is patterned after the MODFLOW input structure, especially the `input structure to MODFLOW-6`_. Larger blocks represent input to MODFLOW packages or modflow-setup features, with sub-blocks representing MODFLOW-6 input blocks (within individual packages) or individual features in modflow-setup. Naming of blocks and the variables within is intended to follow MODFLOW and Flopy naming as closely as possible; where these conflict, the MODFLOW naming conventions are used (see also the `MODFLOW-NWT Online Guide`_).


Package blocks
^^^^^^^^^^^^^^
The modflow-setup configuration file is divided into blocks, which represent sub-dictionaries within the ``cfg`` dictionary represented by the whole configuration file. The blocks are generally organized as input to individual object classes in Flopy, or features specific to MODFLOW-setup. For example, this block would represent input to the `Simulation class`_ for MODFLOW-6:

.. code-block:: yaml

	simulation:
  	  sim_name: 'mfsim'
  	  version: 'mf6'
  	  sim_ws: '../tmp/shellmound'

and would be loaded into the configuration dictionary as:

.. code-block:: python

	cfg['simulation'] = {'sim_name: 'mfsim',
	                     'version': 'mf6',
	                     'sim_ws': '../tmp/shellmound'
	                     }

The above dictionary would then be fed to the Flopy `Simulation class`_ constructor as `keyword arguments (**kwargs)`_.

Sub-blocks
^^^^^^^^^^
Sub-blocks (nested dictionaries) with blocks are used to denote either input to MODFLOW-6 package blocks or input to modflow-setup features. For example, the options block below represents input to the options block for the MODFLOW-6 name file:

.. code-block:: yaml

    model:
      simulation: 'shellmound'
      modelname: 'shellmound'
      options:
        print_input: True
        save_flows: True
        newton: True
        newton_under_relaxation: False
      packages: ['dis',
                 'ic',
                 'npf',
                 'oc',
                 'sto',
                 'rch',
                 'sfr',
                 'obs',
                 'wel',
                 'ims'
                 ]
      external_path: 'external/'
      relative_external_filepaths: True

Note that some items in the model block above do not represent flopy
input. The ``relative_external_filepaths`` item is a flag for modflow-setup that instructs it to reference external files relative to the model workspace, to avoid broken paths when the model is copied to a different location.

Directly specifying MODFLOW input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MODFLOW input can be specified directly in the configuration file using the appropriate variables described in the `MODFLOW-6 input instructions`_ and `MODFLOW-NWT Online Guide`_. For example, in the block below, the dimensions and griddata sub-blocks would be fed directly to the `ModflowGwfdis`_ constructor in Flopy:

.. code-block:: yaml

    dis:
      remake_top: True
      options:
        length_units: 'meters'
      dimensions:
        nlay: 2
        nrow: 30
        ncol: 35
      griddata:
        delr: 1000.
        delc: 1000.
        top: 2.
        botm: [1, 0]


Source_data sub-blocks
^^^^^^^^^^^^^^^^^^^^^^
Alternatively, ``source_data`` subblocks indicate input from general file formats (shapefiles, csvs, rasters, etc.) that needs to be mapped to the model space and time discretization. The ``source_data`` blocks are intended to be general across input types. For example- ``filename`` indicates a file path (string), regardless of the type of file, and ``filenames`` indicates a list or dictionary of files that map to model layers or stress periods. Items with the '_units' suffix indicate the units of the source data, allowing modflow-setup to convert the values to model units accordingly. In the example below, the model top would be read from the specified `GeoTiff`_ and mapped onto the model grid via linear interpolation (the default method for model layer elevations) using the `scipy.interpolate.griddata`_ method. The model botm elevations would be read similarly, with missing layers sub-divided evenly between the specified layers. For example, the layer 7 bottom elevations would be set halfway between the layer 6 and 8 bottoms. Finally, supplying a shapefile as input to idomain instructs modflow-setup to intersect the shapefile with the model grid (using `rasterio.features.rastersize`_), and limit the active cells to the intersected area.

.. code-block:: yaml

    dis:
      remake_top: True
      options:
        length_units: 'meters'
      dimensions:
        nlay: 13
        nrow: 30
        ncol: 35
      griddata:
        delr: 1000.
        delc: 1000.
      source_data:
        top:
          filename: 'shellmound/rasters/meras_100m_dem.tif' # DEM file; path relative to setup script
          elevation_units: 'feet'
        botm:
          filenames:
            0: 'shellmound/rasters/vkbg_surf.tif' # Vicksburg-Jackson Group (top)
            1: 'shellmound/rasters/ucaq_surf.tif' # Upper Claiborne aquifer (top)
            2: 'shellmound/rasters/mccu_surf.tif' # Middle Claiborne confining unit (top)
            3: 'shellmound/rasters/mcaq_surf.tif' # Middle Claiborne aquifer (top)
            6: 'shellmound/rasters/lccu_surf.tif' # Lower Claiborne confining unit (top)
            8: 'shellmound/rasters/lcaq_surf.tif' # Lower Claiborne aquifer (top)
            9: 'shellmound/rasters/mwaq_surf.tif' # Middle Wilcox aquifer (top)
            10: 'shellmound/rasters/lwaq_surf.tif' # Lower Wilcox aquifer (top)
            12: 'shellmound/rasters/mdwy_surf.tif' # Midway confining unit (top)
          elevation_units: 'feet'
        idomain:
          filename: 'shellmound/shps/active_area.shp'


.. _GeoTIFF: https://en.wikipedia.org/wiki/GeoTIFF
.. _input structure to MODFLOW-6: https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.1.0.pdf
.. _keyword arguments (**kwargs): https://stackoverflow.com/questions/1769403/what-is-the-purpose-and-use-of-kwargs
.. _MODFLOW-6 input instructions: https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.1.0.pdf
.. _MODFLOW-NWT Online Guide: https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/
.. _ModflowGwf class: https://github.com/modflowpy/flopy/blob/develop/flopy/mf6/modflow/mfgwf.py
.. _ModflowGwfdis: https://github.com/modflowpy/flopy/blob/develop/flopy/mf6/modflow/mfgwfdis.py
.. _rasterio.features.rastersize: https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
.. _scipy.interpolate.griddata: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
.. _Simulation class: https://github.com/modflowpy/flopy/blob/develop/flopy/mf6/modflow/mfsimulation.py