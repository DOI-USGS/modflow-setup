
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
from .mf6model import MF6model
from .mfnwtmodel import MFnwtModel
from . import gis
from . import interpolate