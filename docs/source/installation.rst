============
Installation
============

``modflow-setup`` depends on a number of python packages, many of which have external C library dependencies. The easiest way to install most of these is with `conda`_. A few packages are not available via conda, and must be installed with `pip`_. If you are on the USGS internal network, see the `Considerations for USGS Users`_ section below first.


See the instructions in `Readme file`_.

Installing python dependencies with Conda
-----------------------------------------
* Download and install the `Anaconda python distribution`_.
* Download an environment file:

  * `environment.yml`_ for a `conda environment`_ with the minimum packages required to run modflow-setup, or
  * `gis.yml`_ for a more full set of packages in the python geospatial stack, including Jupyter Notebooks and packages needed testing, documentation and packaging. Note that the environment described by ``environment.yml`` is called `mfsetup`, while environment in ``gis.yml`` is called `gis`.
  * Alternatively, clone (`using git`_) or `download`_ the ``modflow-setup`` repository, which includes the two environmental files are included at the root level.
  * Note that both of these environment files contain a ``pip`` section of packages that will be installed with pip, after the ``conda`` packages are installed.

Creating a `conda environment`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are on the USGS internal network, see the `Considerations for USGS Users`_ section below first.
Open an Anaconda Command Prompt on Windows or a terminal window on OSX and point it to the location of ``environment.yml`` or ``gis.yml`` and enter:

.. code-block:: bash

    conda env create -f environment.yml

Building the environment will probably take a while. If the build fails because of an SSL error, fix the problem (see `Considerations for USGS Users`_ below) and either:

    a) 	Update the environment

        .. code-block:: bash

            conda env update -f environment.yml

    b) 	or remove and reinstall it:

        .. code-block:: bash

            conda env remove -n mfsetup
            conda env create -f environment.yml

Installing modflow-setup
------------------------
Once a suitable conda environment is made, activate it, so that the version of python in it will be at the top of the system path (i.e. will be the one called when ``python`` is entered at the command line):

.. code-block:: bash

    conda activate mfsetup

Clone or `download`_ the ``modflow-setup`` repository. Then with a command window pointed at the root level (containing ``setup.py``):

.. code-block:: bash

    pip install -e .

This installs modflow-setup to the active python distribution by linking the source code in-place, making it easier to periodically pull updates using git.


Updating modflow-setup using Git
--------------------------------
To update modflow-setup (if it was cloned using Git), at the root level of the repository:

.. code-block:: bash

    git pull origin master

Alternatively, modflow-setup could be updated by downloading the repository again and installing via ``pip install -e .``.


_`Considerations for USGS Users`
--------------------------------
Using conda or pip on the USGS network requires SSL verification, which can cause a number of issues. If you are encountering persistant issues with creating the conda environment, you may have better luck trying the install off of the USGS network (e.g. at home). See `here <https://tst.usgs.gov/applications/application-and-script-signing/>`_ for more information about SSL verification on the USGS network, and to download the DOI SSL certificate.

_`Installing the DOI SSL certificate for use with pip`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1) `Download the DOI SSL certificate`_
2) On Windows, create the file ``C:\Users\<your username>\AppData\Roaming\pip\pip.ini``.
   On OSX, create ``/Users/<your username>/Library/Application Support/pip/pip.conf``.

Include the following in this file:

::

    [global]
    cert = <path to DOI certificate file (e.g. DOIRootCA2.cer)>

Note that when you are off the USGS network, you may have to comment out the ``cert=`` line in the above pip configuration file to get ``pip`` to work.

Installing the DOI SSL certificate for use with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See `these instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#ssl-verification-ssl-verify>`_. This may or may not work.


Troubleshooting issues with the USGS network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**If you are on the USGS network, using Windows, and you get this error message:**

..

    CondaHTTPError: HTTP 500 INTERNAL ERROR for url <https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-gettext-0.19.7-2.tar.bz2>
    Elapsed: 00:30.647993

    An HTTP error occurred when trying to retrieve this URL.
    HTTP errors are often intermittent, and a simple retry will get you on your way.

Adding the following line to ``environment.yml`` should work:

.. code-block:: yaml

    - msys2::m2w64-gettext


This tells conda to fetch ``m2w64-gettext`` from the ``msys2`` channel instead. Note that this is only a dependency on Windows,
so it needs to be commented out on other operating systems (normally it wouldn't need to be listed, but the above HTTP 500 error indicates that installation from the default source location failed.)

**If you are on the USGS network and get an SSL error message**
(something similar to ``SSL: CERTIFICATE_VERIFY_FAILED``), you need to configure the ``pip`` package installer to use the USGS certificate (see `Installing the DOI SSL certificate for use with pip`_ above).



.. _Anaconda python distribution: https://www.anaconda.com/distribution/
.. _conda: https://docs.conda.io/en/latest/
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html
.. _download: https://github.com/aleaf/modflow-setup/archive/master.zip
.. _gis.yml: https://github.com/aleaf/modflow-setup/blob/master/gis.yml
.. _Download the DOI SSL certificate: https://tst.usgs.gov/applications/application-and-script-signing/
.. _pip: https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing
.. _Readme file: https://github.com/aleaf/modflow-setup/blob/master/Readme.md
.. _environment.yml: https://github.com/aleaf/modflow-setup/blob/master/environment.yml
.. _using git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

