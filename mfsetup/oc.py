"""Functions for handling MODFLOW output control
"""
from collections import defaultdict


def parse_oc_period_input(period_input, nstp=None, output_fmt='mf6'):
    """Parse both flopy and mf6-style stress period output control input
    into flopy input.

    Parameters
    ----------
    period_input : dict
        Dictionary of stress period input (see examples)
    nstp : list-like
        Number of timesteps in each stress period
    output_fmt : str
        'mf6' for MODFLOW 6 style input (to :py:func:`flopy.mf6.ModflowGwfoc`), otherwise,
        input for :py:func:`flopy.modflow.ModflowOc` is produced.

    Returns
    -------
    flopy_input : dict
        Input to the flopy output control package constructor.

    Examples
    --------
    >>> period_input = {'saverecord': {0: {'head': 'last', 'budget': 'last'}}
    >>> parse_oc_period_input(period_input)
    {0: [('head', 'last'), ('budget', 'last')]}
    """
    if nstp is not None:
        nstp = list(nstp)
        nper = len(nstp)

    flopy_input = {}
    mf6_flopy_input = {}
    for rec in ['printrecord', 'saverecord']:
        if rec in period_input:
            if output_fmt != 'mf6':
                msg = ("MODFLOW 6 Flopy-style OC input (printrecord or "
                       "saverecord arguments) only supported for MODFLOW 6 models.")
                raise NotImplementedError(msg)
            data = period_input[rec]
            mf6_record_input = {}
            for kper, words in data.items():
                mf6_record_input[kper] = []
                for var, instruction in words.items():
                    mf6_record_input[kper].append((var, instruction))
            mf6_flopy_input[rec] = mf6_record_input
        elif 'period_options' in period_input:
            mf6_record_input = defaultdict(list)
            mf_record_input = defaultdict(list)
            for kper, options in period_input['period_options'].items():
                # empty period for turning off output
                if len(options) == 0:
                    mf6_record_input[kper] = []
                    mf_record_input[(kper, 0)] = []
                else:
                    for words in options:
                        type, var, *instruction = words.split()
                        if type == rec.replace('record', ''):
                            if output_fmt == 'mf6':
                                mf6_record_input[kper].append((var, *instruction))
                            else:
                                if nstp is None:
                                    raise ValueError("MODFLOW 2005-style OC input requires "
                                                        "timestep information.")
                                # parse MF6-style instructions
                                # into MF2005 style input
                                kstp = 0
                                nstep_idx = kper if kper < len(nstp) else -1
                                instruction, *values = instruction
                                instruction = instruction.lower()
                                if instruction == 'all':
                                    for kstp in range(nstp[nstep_idx]):
                                        mf_record_input[(kper, kstp)].append(f"{type} {var}")
                                elif 'frequency' in instruction:
                                    if len(values) == 0:
                                        raise ValueError("mfsetup.oc.parse_oc: "
                                                        "'frequency' instruction needs a value")
                                    freq = int(values[0])
                                    steps = list(range(nstp[nstep_idx]))[::freq]
                                    for kstp in steps:
                                        mf_record_input[(kper, kstp)].append(f"{type} {var}")
                                elif 'steps' in instruction:
                                    if len(values) == 0:
                                        raise ValueError("mfsetup.oc.parse_oc: "
                                                        "'steps' instruction needs one or more values")
                                    for kstp in values:
                                        mf_record_input[(kper, int(kstp))].append(f"{type} {var}")
                                elif instruction == 'first':
                                    mf_record_input[(kper, 0)].append(f"{type} {var}")
                                elif instruction == 'last':
                                    kstp = nstp[nstep_idx] - 1
                                    mf_record_input[(kper, int(kstp))].append(f"{type} {var}")
                                else:
                                    raise ValueError("mfsetup.oc.parse_oc: instruction "
                                                    f"'{instruction}' not understood")
            if len(mf6_record_input) > 0 and output_fmt == 'mf6':
                mf6_flopy_input[rec] = dict(mf6_record_input)
            elif len(mf_record_input) > 0:
                mf_record_input = fill_oc_stress_period_data(mf_record_input, nper=len(nstp))
                flopy_input['stress_period_data'] = dict(mf_record_input)
    if output_fmt == 'mf6':
        return mf6_flopy_input
    return flopy_input


def fill_oc_stress_period_data(stress_period_data, nper):
    """For MODFLOW 2005-style models, repeat last entry in stress_period_data
    for subsequent stress periods (until another entry is encountered),
    as is done by default in MODFLOW 6.
    """
    filled_spd = {}
    last_period_data = {}
    for period in range(nper):
        for (kper, kstp), data in stress_period_data.items():
            if kper == period:
                last_period_data[(kper, kstp)] = data
        last_period_data = {(period, kstp): data for (kper, kstp), data
                            in last_period_data.items()}
        filled_spd.update(last_period_data)
    return filled_spd
