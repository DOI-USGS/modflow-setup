=======================================================================================
Initial Conditions
=======================================================================================

Similar to other packages, input of initial conditions follows the structure of MODFLOW and Flopy. Setting the starting heads from the model top is often a good way to go initially. After the model has been run, starting heads can then be :ref:`updated from the initial model head output <Example script for updating starting heads from a previous run>` to improve convergence on subsequent runs.

    .. Note::

        With any transient model, an :ref:`initial steady-state stress period <Time Discretization>` is recommended, regardless of initial conditions. Specification of initial conditions is mostly a consideration for improving model convergence.

Starting heads defaults
--------------------------
Omitting or leaving the ``ic:`` block blank triggers one of two default actions:

* If there is no parent model, or a parent model with the ``default_source_data`` option turned off, starting heads are set to the model top.
* If there is a parent model with the ``default_source_data`` option turned on, starting heads are set from the parent model starting heads.

Specifying starting heads directly
-----------------------------------
For MODFLOW 6 models, the "Options" and "Griddata" input blocks are represented as sub-blocks within the ``ic:`` block. Within the ``griddata:`` block, model inputs can be specified directly, as long as they are consistent with the definition of the model grid. For example, if ``strt: [100., 100.]`` is specified, the model must be either two layers (for "LAYERED" input; see MODFLOW input instructions) or two cells. An example of specifying a uniform starting head condition for a model of any size:

.. code-block:: yaml

    ic:
      griddata:
        strt: 100.

Or specifying uniform starting heads by layer:

.. code-block:: yaml

    ic:
      griddata:
        strt: [100., 99.]

External files can also be supplied, following the Flopy format:

.. code-block:: yaml

    ic:
      griddata:
        strt:
          - {'filename': 'strt_000.dat'}
          - {'filename': 'strt_001.dat'}

Specifying starting heads from grid-independent sources
--------------------------------------------------------
Similar to other packages, input specified via a ``source_data:`` sub-block can be mapped to the model grid from a variety of sources.

Explicitly setting starting heads to the model top (for example, to override the default behavior with ``default_source_data`` from a parent model):

.. code-block:: yaml

    ic:
      source_data:
        strt: 'from_model_top'

Setting starting heads from a binary head save file (pathed relative to the configuration file):

.. code-block:: yaml

    ic:
      source_data:
        strt:
          from_parent:
            binaryfile: '/path/to/pleasant.hds'
            stress_period: 0

Explicitly setting starting heads to the parent model starting heads (for example, to override the default behavior if ``default_source_data`` is ``False``)

.. code-block:: yaml

    ic:
      source_data:
        strt: from_parent

Example script for updating starting heads from a previous run
---------------------------------------------------------------
The following example script opens MODFLOW binary head results using Flopy, and writes them to external files that can then be input to the IC or BAS6 Packages (for MODFLOW 6 or MODFLOW-NWT models, respectively). If you followed the Modflow-setup defaults to write external files to an ``external/`` folder, with starting head files named as shown in the script, this will overwrite your previous starting head files, requiring no additional action after running the script.

    .. literalinclude:: ../../../examples/update_starting_heads_from_previous.py
        :language: python
        :linenos:

    Download the file:
    :download:`update_starting_heads_from_previous.py <../../../examples/update_starting_heads_from_previous.py>`

**Note:** This script can be run as soon as the initial steady-state stress period is done solving. The current MODFLOW run can then be canceled and MODFLOW restarted, hopefully with improved convergence in the initial period.
