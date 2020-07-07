Configuration defaults
----------------------
The following two yaml files contain default settings for MODFLOW-6 and MODFLOW-NWT. Settings not specified by the user in their configuration file are populated from these files when they are loaded into the ``MF6model`` or ``MFnwtModel`` model instances.

MODFLOW-6 configuration defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. literalinclude:: ../../mfsetup/mf6_defaults.yml
    :language: yaml
    :linenos:

MODFLOW-NWT configuration defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../mfsetup/mfnwt_defaults.yml
    :language: yaml
    :linenos:
