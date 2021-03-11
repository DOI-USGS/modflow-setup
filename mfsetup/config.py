"""
Functions for working with the model configuration dictionary.
"""
import builtins

from mfsetup.fileio import load_cfg
from mfsetup.units import lenuni_text, lenuni_values


def iprint(*args, indent=0, **kwargs):
    args = " "*indent + args[0].replace('\n', '\n'+' '*indent),
    builtins.print(*args, **kwargs)


def validate_configuration(configuration, default_file=None):
    """Validate configuration file by checking for common errors,
    and resolving them if possible.

    Parameters
    ----------
    configuration : str (filepath) or dict

    """
    cfg = configuration
    if not isinstance(cfg, dict):
        cfg = load_cfg(yamlfile, verbose=False, default_file=default_file)

    print('\nvalidating configuration...')
    # DIS package
    print('DIS package')
    if 'length_units' in cfg['dis']:
        if cfg['dis']['length_units'] != lenuni_text[cfg['dis']['lenuni']]:
            iprint((f"length_units: {cfg['dis']['length_units']} "
                   f"but lenuni: {lenuni_text[cfg['dis']['lenuni']]}"), indent=2)
            iprint(f"switching lenuni to {lenuni_values[cfg['dis']['length_units']]}",
                   indent=4)
            cfg['dis']['lenuni'] = lenuni_values[cfg['dis']['length_units']]
    print('done with validation.\n')
