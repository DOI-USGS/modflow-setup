===========================================================
Specifying perimeter boundary conditions from another model
===========================================================

Often the area we are trying to model is part of a larger flow system, and we must account for groundwater flow across the model boundaries. Modflow-setup allows for perimeter boundary conditions to be specified from the groundwater flow solution of another Modflow model.


Features and Limitations
-------------------------
* Currently, specified head perimeter boundaries are supported via the MODFLOW Constant Head (CHD) Package; specified flux boundaries are supported via the MODFLOW Well (WEL) Package.
* The parent model solution (providing the values for the boundaries) is assumed to align with the inset model time discretization.
* The parent model may have different length units.
* The parent model may be of a different MODFLOW version (e.g. MODFLOW 6 inset with a MODFLOW-NWT parent)
* For specified head perimeter boundaries, the inset model grid need not align with the parent model grid; values from the parent model solution are interpolated linearly to the cell centers along the inset model perimeter in the x, y and z directions (using a barycentric triangular method similar to :py:func:`scipy.interpolate.griddata`). However, this means that there may be some mismatch between the parent and inset model solutions along the inset model perimeter, in places where there are abrupt or non-linear head gradients. Boundaries for inset models should always be set sufficiently far away that they do not appreciably impact the model solution in the area(s) of interest. The :ref:`LGR capability <Pleasant Lake test case>` of Modflow-setup can help with this.
* Specified flux boundaries are currently limited to the parent and inset models being colinear.
* The perimeter may be irregular. For example, the edge of the model active area may follow a major surface water feature along the opposite side.
* Specified perimeter heads in MODFLOW-NWT models will have ending heads for each stress period assigned from the starting head of the next stress period (with the last period having the same starting and ending heads). The MODFLOW 6 Constant Head Package only supports assignment of a single head per stress period. This distinction only matters for models where stress periods are subdivided by multiple timesteps.


Configuration input
-------------------
Input to set up perimeter boundaries are specified in two places:

1) The ``parent:`` model block, in which a parent or source model can be specified. Currently only a single parent or source model is supported. The parent or source model can be used for other properties (e.g. hydraulic conductivity) and stresses (e.g. recharge) in addition to the perimeter boundary.

    Input example:

    .. code-block:: yaml

      parent:
        namefile: 'pleasant.nam'
        model_ws: 'data/pleasant/'
        version: 'mfnwt'
        copy_stress_periods: 'all'
        start_date_time: '2012-01-01'
        length_units: 'meters'
        time_units: 'days'

2) In a ``perimeter_boundary:`` sub-block for the relevant package (only specified heads via CHD are currently supported).

    Input example (specified head):

    .. code-block:: yaml

      chd:
        perimeter_boundary:
            parent_head_file: 'data/pleasant/pleasant.hds'

    Input example (specified flux, with optional shapefile defining an irregular perimeter boundary,
    and the MODFLOW 6 binary grid file, which is required for reading the cell budget output from MODFLOW 6 parent models):

    .. code-block:: yaml

      wel:
        perimeter_boundary:
          shapefile: 'shellmound/tmr_parent/gis/irregular_boundary.shp'
          parent_cell_budget_file: 'shellmound/tmr_parent/shellmound.cbc'
          parent_binary_grid_file: 'shellmound/tmr_parent/shellmound.dis.grb'


Specifying the time discretization
------------------------------------
By default, inset model stress period 0 is assumed to align with parent model stress period zero (``copy_stress_periods: 'all'`` in the :ref:`configuration file <The configuration file>` parent block, which is the default). Alternatively, stress periods can be mapped explicitly using a dictionary. For example:

.. code-block:: yaml

  copy_stress_periods:
    0: 1
    1: 2
    2: 3

where ``0: 1`` indicates that the first stress period in the inset model aligns with the second stress period in the parent model (stress period 1), etc.


Specifying the locations of perimeter boundary cells
----------------------------------------------------
Modflow-setup provides 3 primary options for specifying the locations of perimeter cells. In all cases, boundary cells are produced by the :meth:`mfsetup.tmr.TmrNew.get_inset_boundary_cells` method, and the resulting cells (including the boundary faces) can be visualized in a GIS environment with the ``boundary_cells.shp`` shapefile that gets written to the ``tables/`` folder by default.

**1) No specification of where the perimeter boundary should be applied** (e.g. a shapefile) and ``by_layer=False:`` (the default). Perimeter BC cells are applied to active cells that coincide with the edge of the maximum areal footprint of the active model area. In places where the edge of the active area is inside of the max active footprint, no perimeter cells are applied.

    Input example:

    .. code-block:: yaml

      chd:
          perimeter_boundary:
            parent_head_file: 'data/pleasant/pleasant.hds'


**2) No specification of where the perimeter boundary should be applied and ``by_layer=True:``**. This is the same as option 1), but the active footprint is defined by layer from the idomain array. This option is generally not recommended, as it may often lead to boundary cells being included in the model interior (along layer pinch-outs, for example). Users of this option should check the results carefully by inspecting the

    Input example:

    .. code-block:: yaml

      chd:
          perimeter_boundary:
            parent_head_file: 'data/pleasant/pleasant.hds'
            by_layer: True

**3) Specification of perimeter boundary cells with a shapefile**. The locations of perimeter cells can be explicitly specified this way, but they still must coincide with the edge of the active extent in each layer (Modflow-setup will not put perimeter cells in the model interior). (Open) Polyline or Polygon shapefiles can be used; in either case a buffer is used to align the supplied features with the active area edge, which is determined using the :py:func:`Sobel edge detection filter in Scipy <scipy.ndimage.sobel>`.


    Input example:

    .. code-block:: yaml

      chd:
          perimeter_boundary:
            shapefile: 'shellmound/tmr_parent/gis/irregular_boundary.shp'
            parent_head_file: 'shellmound/tmr_parent/shellmound.hds'
