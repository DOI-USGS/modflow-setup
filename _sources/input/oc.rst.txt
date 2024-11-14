===========================================================
MODFLOW Output Control
===========================================================

Stress period input format
--------------------------
Regardless of the model version (MODFLOW-2005-style or MODFLOW 6), output control can be specified in a format similar to native MODFLOW 6 input:

.. code-block:: yaml

    oc:
      period_options:
        0: ['save head last', 'save budget last']
        10: []
        15: ['save head last', 'save budget last']

The above ``period_options:`` block would save the head and cell budget output on the last timestep of stress periods 0 through 9, and from 15 on, but turn off output saving for stress periods 10 through 14. This behavior is consistent with MODFLOW 6 but differs from MODFLOW-2005, where each stress periods must be explicitly included for output to be written. Other options besides ``'last'`` include ``all, first, frequency <frequency>, and steps <steps(<nstp)>``; see the MODFLOW 6 input instructions for more details.

Output filenames and other arguments
------------------------------------
For MODFLOW 6 models, the ``head_fileout_fmt`` and ``budget_fileout_fmt`` arguments can also be supplied to tell Flopy where to save the head and cell budget files, and how to name them. Modflow-setup fills any format specifiers (``'{}'``) with the model name, and passes the resulting strings to the ``head_filerecord`` and ``budget_filerecord`` arguments to the :py:class:`flopy.mf6.ModflowGwfoc <flopy.mf6.modflow.mfgwfoc.ModflowGwfoc>` constructor.

.. code-block:: yaml

    oc:
      head_fileout_fmt: '{}.hds'
      budget_fileout_fmt: '{}.cbc'
      period_options:
        0: ['save head last', 'save budget last']

Any other valid arguments to the :py:class:`flopy.mf6.ModflowGwfoc <flopy.mf6.modflow.mfgwfoc.ModflowGwfoc>` and :py:class:`flopy.modflow.ModflowOc <flopy.modflow.mfoc.ModflowOc>` constructors can be supplied as keys in the ``oc:`` dictionary block. For example:

    .. code-block:: yaml

        oc:
          unitnumber: [14, 51, 52, 53, 0]

would set the unit numbers for the head, drawdown, budget, and ibound output files. See the Flopy documentation for more details. Invalid arguments are filtered out prior to calling the constructor.

Alternative stress period input formats
----------------------------------------
As with other arguments, stress period input can also be directly specified in the Flopy input formats. For example, ``stress_period_data`` could be supplied for a MODFLOW-2005 model as it would be supplied to the :py:class:`flopy.modflow.ModflowOc <flopy.modflow.mfoc.ModflowOc>` constructor:

.. code-block:: yaml

    oc:
      stress_period_data:
        (0, 1): ['save head', 'save budget']
