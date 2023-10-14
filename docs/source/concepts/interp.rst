===========================================================
Interpolating data to the model grid
===========================================================

For most interpolation operations where geo-located data are sampled to the model grid, Modflow-setup uses a barycentric (triangular) interpolation scheme similar to :py:func:`scipy.interpolate.griddata`. This n-dimensional unstructured method allows for interpolation between grids that are aligned with different coordinate references systems, as well as interpolation between unstructured grids. As described `here <https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids>`_, setup of the barycentric interpolation involves:

    1) Construction of a triangular mesh linking the source points
    2) Searching the mesh to find the containing simplex for each destination point
    3) Computation of barycentric coordinates (weights) that describe where each destination point is in terms of the n nearest source points (where n-1 is the number of dimensions)
    4) Computing the interpolated values at the destination points from the source values and the weights

Steps 1-3 are time-consuming. Therefore, for each interpolation problem, Modflow-setup performs these steps once and caches the results, so that step 4 can be repeated quickly on subsequent calls. This can greatly speed, for example, the computation of hydraulic conductivity or bottom elevation values for models with many layers, or interpolation of boundary conditions for models with many stress periods.

A few more notes:
    * Linear interpolation is the default method in most instances, except for recharge, which is often based on categorical data such as land cover and soil types, and therefore has nearest-neighbor as the default method.
    * The interpolation method can generally be specified explicitly for a given dataset by including a ``resample_method`` argument. Available methods are listed in the documentation for :py:func:`scipy.interpolate.griddata`. For example, if we wanted to override the ``'nearest'`` default for the Recharge Package:

    .. literalinclude:: ../../../mfsetup/tests/data/shellmound.yml
        :language: yaml
        :start-after: # Recharge Package
        :end-before: period_stats

    * More details are available in the documentation for the :py:mod:`mfsetup.interpolate` module.
