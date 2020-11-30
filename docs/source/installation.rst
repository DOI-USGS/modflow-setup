============
Installation
============

``Modflow-setup`` depends on a number of python packages, many of which have external C library dependencies. The easiest way to install most of these is with `conda`_. A few packages are not available via conda, and must be installed with `pip`_. If you are on the USGS internal network, see the `Considerations for USGS Users`_ section below first.

Installing python dependencies with Conda
-----------------------------------------
* Download and install the `Anaconda python distribution`_.

  * **Make sure to install Anaconda to your username** (not at the system level). More often than not, installing at the system level (for all users) seems to result in issues with library dependencies (for example, import of ``fiona`` or ``rasterio`` failing because gdal isn't found). It is also good practice to periodically do a `clean uninstall`_ of Anaconda, which at the system level requires admin. privileges.

    * In the installer, at the “Destination Select” step, select “Install for me only.” It should say something about how the software will be installed to your home folder.
    * If your installer skips the “Destination Select” step, when you get to "Installation Type", click “Change Install Location” and then “Install for me only.”


* Download an environment file:

  * `requirements.yml`_ for a `conda environment`_ with the minimum packages required to run Modflow-setup, or
  * `gis.yml`_ for a more full set of packages in the python geospatial stack, including Jupyter Notebooks and packages needed for testing, documentation and packaging. Note that the environment described by ``requirements.yml`` is called `mfsetup`, while the environment in ``gis.yml`` is called `gis`.
  * Alternatively, clone (`using git`_) or `download`_ the ``modflow-setup`` repository, which includes the two environment files at the root level.
  * Note that both of these environment files contain a ``pip`` section of packages that will be installed with pip, after the ``conda`` packages are installed.

Creating a `Conda environment`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are on the USGS internal network, see the `Considerations for USGS Users`_ section below first.
Open an Anaconda Command Prompt on Windows or a terminal window on OSX and point it to the location of ``requirements.yml`` or ``gis.yml`` and enter:

.. code-block:: bash

    conda env create -f requirements.yml

Building the environment will probably take a while. If the build fails because of an SSL error, fix the problem (see `Considerations for USGS Users`_ below) and either:

    a) 	Update the environment

        .. code-block:: bash

            conda env update -f requirements.yml

    b) 	or remove and reinstall it:

        .. code-block:: bash

            conda env remove -n mfsetup
            conda env create -f requirements.yml

Keeping the Conda environment up to date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The python packages and other open source software libraries that Modflow-setup depends on are continually changing. Modflow-setup aims to mostly follow the `Numpy guidelines for package support <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_, which effectively means that the two latest minor versions of Python (e.g. 3.8 and 3.7) and their associated Numpy versions will be supported. However, occasionally backwards compatability with a particular package may be broken in a shorter timeframe, in which case the minimum required version of that package will be specified in the ``requirements.yml`` file. All of this to say that your Conda environment will eventually get out of date. The `Conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ has instructions for updating packages within a Conda environment, but at some point (perhaps a few times a year) it is good practice to simply delete the environment and rebuild it from the `.yml` file. Every so often, you may also want to reinstall Anaconda after a `clean uninstall`_.

Installing Modflow-setup
-----------------------------
There are several ways to install Modflow-setup. Regardless of the method, the installation must be performed in a python
environment with the required dependencies. In the case of the Conda environment created above, the environment must be activated, so that right version of python is called when ``python`` is entered at the command line:

.. code-block:: bash

    conda activate mfsetup


Installing the latest develop version of Modflow-setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pip can be used to fetch Modflow-setup directly from GitHub:

.. code-block:: bash

    pip install git+git://github.com/aleaf/modflow-setup@develop

Subsequent updates can then be made with

.. code-block:: bash

    pip uninstall modflow-setup
    pip install git+git://github.com/aleaf/modflow-setup@develop

Installing the Modflow-setup source code in-place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alternatively, if you intend to contribute to Modflow-setup (please do!) or update your install frequently, the best route is probably to clone the source code from git and install it in place.

.. code-block:: bash

    git clone https://github.com/aleaf/modflow-setup.git
    cd modflow-setup
    pip install -e .

.. note::
    Don't forget the ``.`` after ``pip install -e``!

Your local copy of the Modflow-setup repository can then be subsequently updated with

.. code-block:: bash

    git pull origin master

.. note::
    If you are making local changes to Modflow-setup that you want to contribute, the workflow is slightly different. See the :ref:`Contributing to Modflow-setup` page for more details.


The advantage of installing the source code in-place is that any changes you make are automatically incorporated into your python environment, without any additional install commands. When debugging in an interactive development environment (IDE) such as Pycharm or VS Code, error tracebacks and inspection features go to the actual source code, not the version installed in the ``site-packages`` folder. Additionally, since this install is done through pip, ``pip uninstall``
will work to remove the package, and the current version of the package (including the latest commit information) will be visible with ``conda list``.


_`Considerations for USGS Users`
--------------------------------
Using conda or pip on the USGS network requires SSL verification, which can cause a number of issues.
If you are encountering persistant issues with creating the conda environment,
you may have better luck trying the install off of the USGS network (e.g. at home).
See `here <https://tst.usgs.gov/applications/application-and-script-signing/>`_ for more information
about SSL verification on the USGS network, and to download the DOI SSL certificate.

_`Installing the DOI SSL certificate for use with pip`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1) `Download the DOI SSL certificate`_
2) `Create a pip configuration file <https://pip.pypa.io/en/stable/user_guide/#config-file>`_, which may be specified at the user, environment, or system level. At the user level:

    * On Windows: ``%APPDATA%\pip\pip.ini`` or ``%APPDATA%\Roaming\pip\pip.ini``, where ``%APPDATA%`` is most likely ``C:\Users\<your username>\AppData``
    * On OSX, create ``$HOME/Library/Application Support/pip/pip.conf`` if directory ``$HOME/Library/Application Support/pip`` exists else ``$HOME/.config/pip/pip.conf.``,
      where ``$HOME`` is most likely ``/Users/<your username>``.
    * On Linux: ``$HOME/.config/pip/pip.conf``

For more details about the configuration file, see `the pip documentation <https://pip.pypa.io/en/stable/user_guide/#config-file>`_.

Include the following in this file:

::

    [global]
    cert = <path to DOI certificate file (e.g. DOIRootCA2.cer)>

Note that when you are off the USGS network, you may have to comment out the ``cert=`` line in the above pip configuration file to get ``pip`` to work.

Installing the DOI SSL certificate for use with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See `these instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#ssl-verification-ssl-verify>`_.
This may or may not work. Basically, ``ssl_verify:`` needs to be set in your `condarc`_ file to point
to a valid SSL certificate, which may be different from the basic ``DOIRootCA2.cer`` file.

You can find the location of your `condarc`_ file with::

    conda info -a

which displays information about how Conda is configured. Note that you may have multiple `condarc`_
files at the system, user and possibly project levels.

Common issues:

* Conda Install fails on the USGS network without a certificate, or with an incorrectly formatted certificate.
  Possible solutions are to either get a correctly formatted SSL certificate from your IT person, or try installing off the network.
* Conda Install fails off the USGS network with a certificate (may or may not be correctly formatted). Solution:
  open your `condarc`_ file
  and comment out the SSL certificate file, if it is specified. E.g.::

    ssl_verify: #D:\certificates\DOIRootCA2.cer



Troubleshooting issues with the USGS network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSL-related error messages when using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(with ``SSL`` mentioned in the message and possibly ``bad handshake``)

Make sure that the ``conda`` package installer is configured to use the USGS certificate
(see :ref:`Installing the DOI SSL certificate for use with conda` above).


SSL-related error messages when using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(something similar to ``SSL: CERTIFICATE_VERIFY_FAILED``).

Make sure that the ``pip`` package installer is configured to use the USGS certificate
(see `Installing the DOI SSL certificate for use with pip`_ above).

If you are on the USGS network, using Windows, and you get this error message:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


.. _Anaconda python distribution: https://www.anaconda.com/distribution/
.. _clean uninstall: https://docs.anaconda.com/anaconda/install/uninstall/
.. _conda: https://docs.conda.io/en/latest/
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html
.. _condarc: https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html
.. _download: https://github.com/aleaf/modflow-setup/archive/master.zip
.. _gis.yml: https://github.com/aleaf/modflow-setup/blob/master/gis.yml
.. _Download the DOI SSL certificate: https://tst.usgs.gov/applications/application-and-script-signing/
.. _pip: https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing
.. _Readme file: https://github.com/aleaf/modflow-setup/blob/master/Readme.md
.. _requirements.yml: https://github.com/aleaf/modflow-setup/blob/master/requirements.yml
.. _using git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
