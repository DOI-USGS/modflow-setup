from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
from mfsetup.fileio import load_modelgrid
from mfsetup.mf6model import MF6model
from mfsetup.mfnwtmodel import MFnwtModel

from . import _version, interpolate

__version__ = _version.get_versions()['version']
