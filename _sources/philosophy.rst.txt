Philosophy
==========

Motivation
----------
Mapping disparate data to a computational grid to create a numerical groundwater model is often a cumbersome, labor-intensive process. As a result, numerical groundwater models (such as those created for MODFLOW) are often late, over budget or sub-optimal in their ability to answer the questions that motivated their construction. The inherent difficulty of changing the discretization or model structure makes it difficult to revisit these choices later in a project in response to what is learned. Carrying alternative conceptual models through the modeling process is seldom feasible. At the same time, most of the cognitive load of the groundwater modeler is consumed with tedious data munging tasks, leaving less room to apply their `hydrosense`_ to the problem.

Scripting with a high-level language like python has been proposed as a solution to some of these challenges (Bakker and others, 2016), but in practice, this is easier said than done. Groundwater modeling workflows are often complex, with many interdependencies. For example, changes to the model stream network might require adjustments to the model layer elevations, locations of inactive cells and locations of other boundary conditions. In turn the stream network input (for example, input to the MODFLOW SFR package) is dependent on the model layering and locations of inactive cells. Even simple changes to fundamental inputs such as the model time discretization can require rebuilding most of the model input files. In principle, an ad-hoc scripting approach could successfully overcome these obstacles with extreme discipline and care, but experience has shown that as a project progresses, it becomes increasingly difficult to simply “rebuild the model from scratch.” 

What if often-used data processing routines for creating groundwater flow models could be collected into a single code base that is hardened with an automated test suite, and improved incrementally with collaborative version control? What if interdependencies between packages could be handled in-memory, allowing for real-time updating as the model is changed? Finally, what if the inputs to a groundwater flow model (both external files and model settings) could be succinctly summarized in a single configuration file that allows the model to be assembled from scratch with a single line of code? What if this allowed us to more easily do `step-wise modeling`_ (Haitjema, 2005) with numerical models?

What modflow-setup does
-----------------------
Modflow-setup aims to distill common operations for constructing MODFLOW models into a single, tested codebase, allowing them to be reused reliably across different projects. Source data can include shapefiles, rasters, NetCDF files and other MODFLOW models that are geo-located. Modflow-setup extends the datatypes used by the `Flopy package`_ (Bakker and others, 2016) to facilitate reading and writing of MODFLOW package input, and handle inter-package dependencies in-memory.  Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. In a few minutes, an external array-based flopy model instance with the desired packages is created from the sampled source data and default settings. MODFLOW input can then be written from the flopy model instance. Both MODFLOW-6 and MODFLOW-NWT are supported.

What modflow-setup doesn’t do
-----------------------------
While modflow-setup strives to be general, it is focused on producing groundwater flow models at the site to regional scales. Furthermore, current development has been focused around project needs; as opposed to making a comprehensive tool from the ground up. Supported packages and features are summarized in the Release History.

Currently only regular (unrotated) model grids are supported; support for unstructured grids may be added in the future. In contrast to Flopy, which is completely general, modflow-setup limits model construction options somewhat, in the interest of rapidly producing consistent results with a minimum of required input. For example, models are produced with the external files option in flopy, meaning that most array and table input are written to external files in a single folder. This facilitates parameter estimation and allows most of the intensive i/o operations to be performed by pandas and numpy, which are generally faster than parsing MODFLOW package input with base python.


References
^^^^^^^^^^
Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.

Haitjema, H.M. (1995). Analytic Element Modeling of Groundwater Flow. Academic Press, Inc.

Hunt, R.J. and Zheng, C. (2012), The Current State of Modeling. Groundwater, 50: 330-333. doi:10.1111/j.1745-6584.2012.00936.x


.. _hydrosense: https://ngwa.onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-6584.2012.00936.x

.. _step-wise modeling: https://www.haitjema.com/stepwise.html

.. _Flopy package: https://github.com/modflowpy/flopy

