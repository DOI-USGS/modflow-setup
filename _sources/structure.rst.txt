Basic program structure and usage
=================================

Modflow-setup is intended to provide both a collection of python modules with functions for performing various model construction tasks and a boiler plate workflow for automated construction of a working MODFLOW model from a configuration file. The latter goal is implemented in the ``MF6model`` and ``MFnwtModel`` classes, which can both be imported at the top level.

.. code-block:: python

    from mfsetup import MF6model
    from mfsetup import MFnwtModel


MF6model extends the ``mf6.ModflowGwf`` (MODFLOW-6 groundwater flow model) class in Flopy; MFnwtModel extends the ``modflow.Modflow`` (pre-MODFLOW-6 groundwater flow model) class in Flopy. These two classes are intended to provide methods and attributes specific to MODFLOW-6 or MODFLOW-NWT. A third `mixin class`_ (``MFsetupMixin``) contains methods and attributes that are general to both MODFLOW-6 and MODFLOW-NWT (and therefore used by both MF6model and MFnwtModel). The goal in this is to make modflow-setup be agnostic to MODFLOW version to the extent feasible.

The above three model classes extend Flopy with methods and property attributes that perform various model construction tasks, such as creating a model grid, creating arrays from source data, managing dependencies between packages, and setting up individual packages. Information for creating the model is stored in a python dictionary under the attribute ``cfg``, which is attached to the model classes. The ``cfg`` dictionary is populated from a configuration file containing all of the information needed to build the model. More about the configuration file in the configuration file section.

Setting up a full model
-----------------------
Instances of ``MF6model`` and ``MFnwtModel`` can be created by setting up full model from a configuration file:

.. code-block:: python

    model = MF6model.setup_from_yaml('config_file.yml')


Loading a model
---------------
Instances of ``MF6model`` and ``MFnwtModel`` can also be created by loading an existing model that has a configuration file (usually this is a model created by modflow-setup):

.. code-block:: python

    model = MF6model.load('config_file.yml')

In the latter case, the model is simply loaded with flopy and the ``MF6model`` instance created with the ``cfg`` dictionary populated from the configuration file, as well as other attributes specific to modflow-setup.

Setting up individual packages
------------------------------
The ``.setup_from_yaml`` method above calls individual package setup methods for all of the packages specified in the configuration file. Alternatively, these package setup methods (with contain the suffix "setup") can be called individually to make or remake a package.

.. code-block:: python

    model = MF6model.load('config_file.yml')
    model.setup_dis()

In this case, the discretization package is built from information in `config_file.yml`.


.. _mixin class: https://en.wikipedia.org/wiki/Mixin
