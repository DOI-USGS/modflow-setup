===========================================================
Local grid refinement
===========================================================

In MODFLOW 6, two groundwater models can be tightly coupled in the same simulation, which allows for efficient "local grid refinement" (LGR; Mehl and others, 2006) and "semistructured" (Feinstein and others, 2016) configurations that combine multiple structured model layers at different resolutions (Langevin and others, 2017). Locally refined area(s) are conceptualized as separate model(s) that are linked to the surrounding (usually coarser) "parent" model through the GWF Model Exchange Package. Similarly, "semistructured" configurations are represented by multiple linked models for each layer or group of layers with the same resolution.

Modflow-setup supports this functionality via an ``lgr:`` subblock within the ``setup_grid:`` block of the configuration file. The ``lgr:`` subblock consists of one of more subblocks, each keyed by a linked model name and containing configuration input for that linked model. Vertical layer refinement relative to the "parent" model, and the range of parent model layers that will be occupied by the linked "inset" model are also specified.

For example, the following "parent" configuration for the :ref:`Pleasant Lake Example <Pleasant Lake test case>` creates a local refinement model named "``pleasant_lgr_inset``" that spans layers 0 through 3 (the first 4 layers) of the parent model, at the same vertical resolution (1 inset model layer per parent model layer). The horizontal location and resolution of the inset model are specified in the inset model configuration file, in the same way that they are specified for any model.

.. literalinclude:: ../../../examples/pleasant_lgr_parent.yml
    :language: yaml
    :start-at: lgr:
    :end-before: # Structured Discretization Package

Input from the ``lgr:`` subblock and the inset model configuration file(s) is passed to the  :py:class:`Flopy Lgr Utility <flopy.utils.lgrutil.Lgr>`, which helps create input for the GWF Model Exchange Package.

Within the context of a Python session, inset model information is stored in a dictionary under an ``inset`` attribute attached to the parent model. For example, to access a Flopy model object for the above inset model from a parent model named ``model``:

.. code-block:: python

	inset_model = model.inset['pleasant_lgr_inset']


**A few notes about LGR functionality in Modflow-setup**

* **Locally refined "inset" models must be aligned with the parent model grid**, which also means that their horizontal resolution must be a factor of the "parent" model resolution. Modflow-setup handles the alignment automatically by "snapping" inset model grids to the lower left corner of the parent cell containing the lower left corner of the inset model grid (the inset model origin in real-world coordinates).
* Similarly, inset models need to align vertically with the parent model layers. Parent layers can be subdivided using the ``layer_refinement:`` input option.
* In principal, a **semistructured** configuration consisting of a "parent" model at the coarsest horizontal resolution, and one or more inset models representing [horizontally and potentially vertically] refined layers within the "parent" model domain is possible, but this configuration has not been tested with Modflow-setup. Please contact the Modflow-setup development team by posting a `GitHub issue <https://github.com/DOI-USGS/modflow-setup/issues>`_ if you are interested in this.
* Similarly, multiple inset models at different horizontal locations, and even inset models within inset models should be achievable, but have not been tested.
* **Multi-model configurations come with costs.** Each model within a MODFLOW 6 simulation carries its own set of files, including external text array and list input files to packages. As with a single model, when using the automated parameterization functionality in `pyEMU <https://github.com/pypest/pyemu>`_, the number of files is multiplied. At some point, at may be more efficient to work with individual models, and design the grids in such a way that boundary conditions along the model perimeters have a minimal impact on the predictions of interest.
