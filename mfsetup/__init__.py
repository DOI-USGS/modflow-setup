from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
from . import interpolate
from .fileio import load_modelgrid
from .mf6model import MF6model
from .mfnwtmodel import MFnwtModel
