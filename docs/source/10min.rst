10 Minutes to Modflow-setup
============================
This is a short introduction to help get you up and running with Modflow-setup. A complete workflow can be found in the :ref:`Pleasant Lake Example`; additional examples of working configuration files can be found in the :ref:`Configuration File Gallery`.

1) Define the model active area and coordinate reference system
-----------------------------------------------------------------
Depending on the problem, the model area might simply be a box enclosing features of interest and any relevant hydrologic boundaries, or an irregular shape surrounding a watershed or other feature. In either case, it may be helpful to :ref:`download hydrography first <3) Develop flowlines to represent streams>`, to ensure that the model area includes all important features. The model should be referenced to a `projected coordinate reference system (CRS) <https://en.wikipedia.org/wiki/Projected_coordinate_system>`_, ideally with length units of meters and an authority code (such as an `EPSG code <https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset>`_) that unambiguously defines it.

Modflow-setup provides two ways to define a model grid:

    * x and y coordinates of the model origin (lower left or upper left corner), grid spacing, number of rows and columns, rotation, and CRS
    * As a rectangular area of specified discretization surrounding a polygon shapefile of the model active area (traced by hand or developed by some other means) or a feature of interest buffered by a specified distance.

The active model area is defined subsequently in the DIS package.

    .. Note::

        Don't forget about the farfield! Usually it is advised to include important competing sinks outside of the immediate area of interest   (the nearfield), so that the solution is not over-specified by the perimeter boundary condition, and recognizing that the surface watershed boundary doesn't always coincide exactly with the groundwatershed boundary. See Haitjema (1995) and Anderson and others   (2015) for more info.

    .. Note::
        Need a polygon defining a watershed? In the United States, the `Watershed Boundary Dataset <https://www.usgs.gov/national-hydrography/access-national-hydrography-products>`_ provides watershed deliniations at various scales.


2) Create a setup script and configuration file
------------------------------------------------
Usually creating the desired grid requires some iteration. We can get started on this by making a model setup script and corresponding configuration file.

An initial model setup script for making the model grid:

    .. literalinclude:: ../../examples/initial_grid_setup.py
        :language: python
        :linenos:

    Download the file:
    :download:`initial_grid_setup.py <../../examples/initial_grid_setup.py>`

An initial configuration file for developing a model grid around a pre-defined active area:

    .. literalinclude:: ../../examples/initial_config_poly.yaml
        :language: yaml
        :linenos:

    Download the file:
    :download:`initial_config_poly.yaml <../../examples/initial_config_poly.yaml>`

To define a model grid using an origin, grid spacing and dimensions, a ``setup_grid:`` block like this one could be substitued above:

    .. literalinclude:: ../../examples/initial_config_box.yaml
        :language: yaml
        :start-at: setup_grid:

    Download the file:
    :download:`initial_config_poly.yaml <../../examples/initial_config_box.yaml>`

Now ``initial_setup_script.py`` can be run repeatedly to explore different grids.


3) Develop flowlines to represent streams
------------------------------------------
Next, let's get some data for setting up boundary conditions. For streams, Modflow-setup can accept any linestring shapefile that has a routing column indicating how the lines connect to one another. This can be created by hand, or in the United States, obtained from the National Hydrography Dataset Plus (NHDPlus). There are two types of NHDPlus:

    - `NHDPlus version 2 <https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus>`_ is mapped at the 1:100,000 scale, and is therefore suitable for larger regional models with cell sizes of ~100s of meters to ~1km. NHDPlus version 2 can be the best choice for larger model areas (greater than approx 1,000 km\ :sup:`2`), where NHDPlus HR might have too many lines. NHDPlus version 2 can be obtained from the `EPA <https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus>`_.
    - `NHDPlus High Resolution (HR) <https://www.usgs.gov/national-hydrography/nhdplus-high-resolution>`_ is mapped at the finer 1:24,000 scale, and may therefore work better for smaller problems (discretizations of ~100 meters or less) where better alignment between the mapped lines and stream channel in the DEM is desired, and where the number of linestring features to manage won't be prohibitive. NHDPlus HR can be accessed via the `National Map Downloader <https://apps.nationalmap.gov/downloader/>`_.

Preprocessing NHDPlus HR
^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, NHDPlus HR data, which comes in a file geodatabase (GDB), must be preprocessed into a shapefile for input to Modflow-setup and `SFRmaker <https://github.com/DOI-USGS/sfrmaker>`_ (which Modflow-setup uses to build the stream network). In many cases, multiple GDBs may need to be combined and undesired line features such as storm sewers culled. The `SFRmaker documentation <https://doi-usgs.github.io/sfrmaker/latest/index.html>`_ has examples for how to read and preprocesses NHDPlus HR.

Preprocessing NHDPlus version 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Depending on the application, NHDPlus version 2 may not need to be preprocessed. Reasons to preprocess include:

* the model area is large, and

    * read times for one or more NHDPlus drainage basins are slowing the model build
    * the DEM being used for the model top is relatively coarse, and sampling a fine DEM during the model build is prohibitive for time or space reasons.

* the stream network is too dense, with too many model cells containing SFR reaches (especially a problem in the eastern US at the 1 km resolution); or there are too many ephemeral streams represented.
* the stream network has divergences where one or more distributary lines are downstream of a confluence.

The `preprocessing module in SFRmaker <https://doi-usgs.github.io/sfrmaker/latest/notebooks/preprocessing_demo.html>`_ can resolve these issues, producing a single set of culled flowlines with width and elevation information and divergences removed. The elevation functionality in the preprocessing module requires a DEM.


4) Get a DEM
-------------
The `National Map Downloader <https://apps.nationalmap.gov/downloader/>`_ has 10 meter DEMs for the United States, with finer resolutions available in many areas. Typically, these come in 1 degree x 1 degree tiles. If many tiles are needed, the uGet Download Manager linked to on the National Map site can automate downloading many tiles. Alternatively, links to the files follow a consistent format, and are therefore amenable to scripted or manual downloads. For example, the tile located between -88 and -87 west and 43 and 44 north is available at:

https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n44w088/USGS_13_n44w088.tif

Making a virtual raster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once all of the tiles are downloaded, a virtual raster can be made that allows them to be treated as a single file, without any modifications to the original data. This is required for input to SFRmaker and Modflow-setup. For example, in `QGIS <https://qgis.org/>`_:

    a) Load all of the tiles to verify that they are correct and cover the whole model active area.
    b) From the ``Raster`` menu, select ``Miscellaneous > Build Virtual Raster``. This will make a virtual raster file with a ``.vrt`` extension that points to the original set of GeoTIFFs, but allows them to be treated as a single continuous raster.

5) Make a minimum working configuration file and model build script
--------------------------------------------------------------------
Now that we have a set of flowlines and a DEM (and perhaps shapefiles for other surface water boundaries), we can fill out the rest of the configuration file to get an initial working model. Later, additional details such as more layers, a well package, observations, or other features can be added in a stepwise approach (Haitjema, 1995).

    .. literalinclude:: ../../examples/initial_config_full.yaml
        :language: yaml
        :linenos:

    Download the file:
    :download:`initial_config_full.yaml <../../examples/initial_config_full.yaml>`

A setup script for making a minimum working model. Additional functions can be added later to further customize the model outside of the Modflow-setup build step.

    .. literalinclude:: ../../examples/initial_model_setup.py
        :language: python
        :linenos:

    Download the file:
    :download:`initial_model_setup.py <../../examples/initial_model_setup.py>`
