.. Packaging Scientific Python documentation master file, created by
   sphinx-quickstart on Thu Jun 28 12:35:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
modflow-setup |version|
=======================


Modflow-setup is a Python package for automating the setup of MODFLOW groundwater models from grid-independent source data including shapefiles, rasters, and other MODFLOW models that are geo-located. Input data and model construction options are summarized in a single configuration file. Source data are read from their native formats and mapped to a regular finite difference grid specified in the configuration file. An external array-based `Flopy <https://github.com/modflowpy/flopy>`_ model instance with the desired packages is created from the sampled source data and configuration settings. MODFLOW input can then be written from the flopy model instance.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

	Philosophy <philosophy>
    Installation <installation>
    10 Minutes to Modflow-setup <10min>
    Examples <examples>
    Configuration File Gallery <config-file-gallery>


.. toctree::
  :maxdepth: 2
  :caption: User Guide

   Basic program structure and usage <structure>
   The configuration file <config-file>
   Concepts and methods <concepts/index.rst>
   Input instructions by package <input/index.rst>
   Troubleshooting <troubleshooting>


.. toctree::
  :maxdepth: 1
  :caption: Reference

   Code reference <api/index>
   Configuration file defaults <config-file-defaults>
   Release History <release-history>
   Contributing to modflow-setup <contributing>

.. toctree::
  :maxdepth: 1
  :caption: Bibliography

   References cited <references>
