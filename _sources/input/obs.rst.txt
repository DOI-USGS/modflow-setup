=======================================================================================
Head observations
=======================================================================================

Head observations are set up from one or more CSV tables of observation
locations, specified in the ``source_data:`` sub-block of the ``obs:`` block
(MODFLOW 6) or the ``hyd:`` block (MODFLOW-NWT). Each observation location is
mapped to the model grid from its x, y coordinates. The columns used for the
x and y coordinates, observation names, and observation type can be configured
via the ``source_data:`` block (see the configuration defaults).

By default, each head observation is created in **every** model layer at its x, 
y location (active cells only). The model output can then be post-processed to 
obtain a simulated head at each observation's actual open interval. Where the 
open interval spans more than one layer, the head can be computed as a 
transmissivity-weighted average of the heads in those layers.

Specifying the layer(s) for head observations
-----------------------------------------------
If the open interval (screened layer) of an observation is known, the
observation can be restricted to one or more specific layers using a layer
column in the source data. The column name is configured with the
``layer_col`` option (default: ``'layer'``). 

The value in the layer column for a given observation may be either:

* a single (zero-based) layer index, for example ``2``, or
* a list of layer indices, for example ``0, 2`` (comma- or space-separated
  in a CSV cell), to place the observation in multiple layers.

Observations with no value in the layer column (missing or empty) fall back to
the default behavior of being placed in all layers.

For example, given a ``head_targets.csv`` file with a ``layer`` column:

.. code-block:: yaml

    obs:
      source_data:
        filenames:
          - 'data/head_targets.csv'
        column_mappings:
          obsname: ['obsprefix', 'common_name']
        x_location_col: 'x'
        y_location_col: 'y'
        layer_col: 'layer'  # optional; 'layer' is the default
