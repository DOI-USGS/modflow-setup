=======================================================================================
Time and space discretization
=======================================================================================

This page describes spatial and temporal discretization input options to the Discretization (DIS) and Time Discretization (TDIS) Packages. Specification of the model active area in the DIS Package (MODFLOW 6) and BAS6 Package (MODFLOW-2005/NWT) is also covered. As always, additional input examples can be found in the :ref:`Configuration File Gallery` and :ref:`Configuration defaults` pages.

As stated previously, a key paradigm of Modflow-setup is setup of space and time discretization during the automated model build, from grid-independent inputs. This allows different discretization schemes to be readily tested without extensive modifications to the inputs.

Spatial Discretization
----------------------
Similar to other packages, input to the Discretization Package follows the structure of MODFLOW and Flopy. For MODFLOW 6 models, the "Options", "Dimensions" and "Griddata" input blocks are represented as sub-blocks within the ``dis:`` block. Within these blocks, model inputs can be specified directly, as long as they are consistent with the definition of the model grid. For example, if ``nlay: 2`` is specified, then the model bottom must be specified as two scalar values, or two ``nrow`` x ``ncol`` arrays:

.. code-block:: yaml

    dis:
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

More commonly, only ``delr`` and ``delc`` are specified in the ``griddata:`` block, and geolocated, grid-independent raster surfaces are supplied in a ``source_data`` sub-block. Modflow-setup then interpolates values from these surfaces to the grid cell centers.

.. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
      :language: yaml
      :start-after: Discretization Package
      :end-before: # Temporal Discretization Package

**A few notes:**

   * by default, linear interpolation is used, as described :ref:`here <Interpolating data to the model grid>`.
   * If a more sophisticated sampling strategy is desired, for example computing mean elevations with zonal statistics for the model top, the respective layers should be pre-processed prior to input to Modflow-setup. This is by design, as it avoids adding additional complexity to the Modflow-setup codebase, and expensive operations like zonal statistics can greatly slow a model build time and often only need to be done infrequently (in contrast to other changes where rapid iteration may be helpful).
   * GeoTIFFs are generally best, because they include complete projection information (including the coordinate reference system) and generally use less disk space than other raster types.
   * if an ``elevation_units:`` item is included, elevation values in the rasters will be converted to the model units
   * the most straightforward way to input layer elevations is to simply assign a raster surface to each layer:

   .. code-block:: yaml

      botm:
        filenames:
          0: bottom_of_layer_0.tif
          1: bottom_of_layer_1.tif
          ...

   * Alternatively, multiple model layers can be inserted between key layer surfaces by simply skipping those numbers. In this exampmle, Modflow-setup creates three layers of equal thickness between the two specified surfaces:

   .. code-block:: yaml

      botm:
        filenames:
          0: bottom_of_layer_0.tif
          # layer 1 bottom is created by Modflow-setup
          # layer 2 bottom is created by Modflow-setup
          3: bottom_of_layer_3.tif
          ...

Adopting layering from a parent model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similar to other input, layer bottoms can be resampled from a parent model. This ``source_data:`` block would simply adopt the same layering scheme as the parent model:

.. code-block:: yaml

  source_data:
    top: from_parent
    botm: from_parent

The parent model layering can also be subdivided by mapping pairs of ``inset: parent`` model layers using a dictionary (YAML sub-block):

.. code-block:: yaml

  source_data:
    top: from_parent
    botm:
      from_parent:
        0: -0.5 # bottom of layer zero in pfl_nwt is positioned at half the thickness of parent layer 1
        1: 0 # bottom of layer 1 corresponds to bottom of layer 0 in parent
        2: 1
        3: 2
        4: 3

In this case, the top layer of the parent model is subdivded into two layers in the inset model. A negative number is used on the parent model side because layer 0 (the first layer bottom) of the parent model coincides with the second layer bottom of the inset model (layer 1). A value of ``-0.5`` places the first inset model layer bottom at half the thickness of the parent model layer; different values between ``-1.`` and ``0.`` could be used to move this surface up or down within the parent model layer, or multiple inset model layers could be specified within the first parent model layer:

.. code-block:: yaml

  source_data:
    top: from_parent
    botm:
      from_parent:
        0: -0.9 # bottom of layer 1 set at 10% of the depth of layer 0 in the parent
        1: -0.3 # bottom of layer 2 set at 70% of the depth of layer 0 in the parent
        2: 0
        3: 1
        4: 2

MODFLOW-2005/NTW input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specification of ``source_data:`` blocks is the same for MODFLOW 6 and MODFLOW-2005 style models, except the latter wouldn't contain an ``idomain:`` subblock. Specification of other inputs generally follows Flopy (for example, :py:class:`~flopy.modflow.mfdis.ModflowDis`). A ``dis:`` block equivalent to the first example give would look like:

.. code-block:: yaml

    dis:
      length_units: 'meters'
      nlay: 2
      nrow: 30
      ncol: 35
      delr: 1000.
      delc: 1000.
      top: 2.
      botm: [1, 0]

.. note::
   The ``length_units:`` item is specific to Modflow-setup; in a MODFLOW-2005 context, Modflow-setup takes this input and enters the appropriate value of ``lenuni`` to Flopy (which writes the MODFLOW input).

Modflow-setup specific input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``drop_thin_cells``: Option in MODFLOW 6 models to remove cells less than a minimum layer thickness from the model solution.
* ``minimum_layer_thickness:`` Minimum layer thickness to allow in model. In MODFLOW 6 models, if ``drop_thin_cells: True``, layers thinner than this will be collapsed to zero thickness, and their cells either made inactive (``idomain=0``) or, if they are between two layers greater than the minimum thickness, converted to vertical pass-through cells (``idomain=1``). In MODFLOW-2005 models or if ``drop_thin_cells: False``, thin layers will be expanded downward to the minimum thicknesses.

Time Discretization
----------------------
In MODFLOW 6, time discretization is specified at the simulation level, it its own Time Discretization (TDIS) Package. In MODFLOW-2005/NWT, time discretization is specified in the Discretization Package. Accordingly, in Modflow-setup, time discretization in specified in the appropriate package block for the model version.

Specifying stress period information directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input to the DIS and TDIS packages follows the MODFLOW structure. For simple steady-state models, time discretization could be specified directly to the DIS or TDIS packages using there respective the Flopy inputs (:py:class:`~flopy.modflow.mfdis.ModflowDis`; :py:class:`~flopy.mf6.modflow.mftdis.ModflowTdis`). This example from the :ref:`Configuration File Gallery` shows direct specification of stress period information to the Discretization Package:

.. literalinclude:: ../../../mfsetup/tests/data/pfl_nwt_test.yml
      :language: yaml
      :start-after: arguments to flopy.modflow.ModflowDis
      :end-before: bas6:

Specifying uniform stress periods frequencies by group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For transient models, we often want to combine an initial steady-state period with subsequent transient periods, which may be of variable lengths. To facilitate this, Modflow-setup has a ``perioddata:`` sub-block that can in turn contain multiple sub-blocks representing stress period "groups". Each group in the ``perioddata:`` sub-block contains information to generate one or more stress periods at a specified frequency and time datum (for example, months, days, every 7 days, etc.). Input to transient groups is based on the :py:func:`pandas.date_range` function, where three of the four ``start_date_time``, ``end_date_time``, ``freq`` and ``nper`` parameters must be defined. For example, this sequence of blocks from the :ref:`Configuration File Gallery` generates an initial steady-state period, followed by a 9 year "spin-up" period between two dates, and then biannual stress periods spanning another specified set of sets. Time-step information is also specified, using the MODFLOW variable names.

.. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
      :language: yaml
      :start-after: # Temporal Discretization Package
      :end-before: # Initial Conditions Package

The ``perioddata:`` sub-block can be used within a ``tdis:`` block for MODFLOW 6 models, or a ``dis:`` block for MODFLOW-2005 style models.

Specifying pre-defined stress periods from a CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some model applications, irregular stress periods may be needed that would require many groups to be specifed using the above ``perioddata:`` sub-block. In these cases, a stress period data table can be pre-defined and input as a CSV file:

.. literalinclude:: ../../../mfsetup/tests/data/shellmound_tmr_inset.yml
      :language: yaml
      :start-after: drop_thin_cells: True
      :end-before: sfr:

An example of a valid table is shown below. Note that only the columns listed in the above ``csvfile:`` block are actually needed. ``perlen`` and ``time`` are calculated internally by Modflow-setup; output control (``oc``) can be specified here or in the ``oc:`` package block.

.. csv-table:: Example Stress period data
   :file: ../../../mfsetup/tests/data/shellmound/tmr_parent/tables/stress_period_data.csv
   :header-rows: 1
