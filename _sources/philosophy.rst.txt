Philosophy
==========

Motivation
----------
Mapping disparate data to a computational model grid is often a cumbersome, labor-intensive process. As a result, numerical groundwater models (such as those created for MODFLOW) are often late, over budget or sub-optimal in their ability to answer the questions they were built for. The inherent difficulty of changing the discretization or model structure makes it difficult to revisit these choices later in a project in response to what is learned. Carrying alternative conceptual models through the modeling process is seldom feasible. At the same time, most of the cognitive load of the groundwater modeler is consumed with tedious data munging tasks, leaving less room for `hydrosense`_.

Scripting with a high-level language like python has been `proposed as a solution to some of these challenges <_https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12413>`_, but in practice, this is easier said than done. Groundwater modeling workflows are often complex, with many interdependencies. For example, changes to the model stream network might require adjustments to the model layer elevations, locations of inactive cells and locations of other boundary conditions. In turn the stream network input (for example, input to the MODFLOW SFR package) may depend on the model layering and locations of inactive cells. Even simple changes to fundamental inputs such as the model time discretization can require rebuilding most of the model input files. In principle, an ad-hoc scripting approach could overcome these obstacles with extreme discipline and care, but experience has shown that as a project progresses, entropy sets in and it becomes increasingly difficult to simply “rebuild the model from scratch.”

What if often-used data processing routines for creating groundwater flow models could be collected into a single code base that is hardened with an automated test suite, and improved incrementally with collaborative version control? What if interdependencies between packages could be handled in-memory, allowing for real-time updating as the model is changed? Finally, what if the inputs to a groundwater flow model (both external files and model settings) could be succinctly summarized in a single configuration file that allows the model to be assembled from scratch with a single line of code? What if this allowed us to more easily do `step-wise modeling`_ with numerical models?

What modflow-setup does
-----------------------
Modflow-setup aims to distill common operations for constructing MODFLOW models into a single, tested codebase, allowing them to be reused reliably across different projects. Source data can include shapefiles, rasters, NetCDF files and other MODFLOW models that are geo-located. Modflow-setup extends the datatypes used by the `Flopy package`_ to facilitate reading and writing of MODFLOW package input, and handle inter-package dependencies in-memory.  Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. In a few minutes, an external array-based flopy model instance with the desired packages is created from the sampled source data and default settings. MODFLOW input can then be written from the flopy model instance. Both MODFLOW-6 and MODFLOW-NWT are supported.

What modflow-setup doesn’t do
-----------------------------
While modflow-setup aims to be general, it is focused on producing groundwater flow models at the site to regional scales. Streams are represented using the SFR package; Lakes can be represented using the Lake Package or `high hydraulic conductivity values <https://ngwa.onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-6584.2002.tb02496.x>`_. Current development has been focused around project needs instead of making a comprehensive tool from the ground up. Currently supported packages and features are summarized in the `Release History`_. Examples of valid configuration files used in the test suite can be found in the `Configuration File Gallery`_.

In contrast to Flopy, which is completely general, modflow-setup limits model construction options somewhat. For example,

* models are only produced with the external files option in flopy, meaning that most array and table input are written to external files in a single folder. This facilitates parameter estimation and allows most of the intensive i/o operations to be performed by pandas and numpy, which are generally faster than parsing MODFLOW package input with base python.
* Currently Modflow-setup favors a linear workflow, in which the entire model is (re)built after edits are made to the configuration file or input data. For many models, this process only takes a few minutes or less. While it is possible to :ref:`load a model <Loading a model>`, repeated loading and (re)writing of MODFLOW input with ad hoc edits (as one might do with Flopy, for example) is not well tested. A great way to speed development of a model is to rebuild the model successively with only the packages needed to troubleshoot or test a particular aspect. This can be done by commenting out or removing packages from the ``packages:`` list in the ``model:`` block of the configuration file. When everything works, a full build can then be made with all of the packages included.
* Currently only regular model grids are supported; support for unstructured grids may be added in the future.





.. _Configuration File Gallery: https://doi-usgs.github.io/modflow-setup/docs/build/html/examples.html#configuration-file-gallery
.. _Release History: https://doi-usgs.github.io/modflow-setup/release-history.html
.. _hydrosense: https://ngwa.onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-6584.2012.00936.x

.. _step-wise modeling: https://www.haitjema.com/stepwise.html

.. _Flopy package: https://github.com/modflowpy/flopy
