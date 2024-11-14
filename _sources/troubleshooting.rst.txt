Troubleshooting
================

My script doesn't run
----------------------
Before filing a :ref:`bug report <Bug reports and enhancement requests>`, check these possible solutions:

Updating your python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Check the versions of python and the various package dependencies (especially Numpy) in the environment that you are using with ``conda list``.

* Is your python version more than 1 minor release out of date (e.g. 3.7 if the current version is 3.9)?
* Has it been a year or more since your conda environment was created or updated?

**It might be time for a fresh environment.** Modflow-setup mostly follows the `Numpy guidelines for package support <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_, which means that the two latest minor versions of Python (e.g. 3.9 and 3.8) and their associated Numpy versions will be supported.

:ref:`Instructions for updating your conda environment <Keeping the Conda environment up to date>`

Updating Modflow-setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``conda list`` can also be used to check if you Modflow-setup is up to date.

* if you installed Modflow-setup from GitHub and you need to incorporate a recently pushed bug fix: ``pip install --upgrade git+https://github.com/aleaf/modflow-setup@develop``
* if you cloned the Modflow-setup source code and need to incorporate a recently pushed bug fix, you will need to pull (``git pull``) and possibly reinstall (``pip install --upgrade -e .`` in the root folder of the source code).
* in both cases, the version reported by ``conda list`` should match the latest commit hash. For example version ``0.post250.dev0+g9af1c61`` for the commit hash starting with ``9af1c61``.
