import numpy as np
from .lakes import make_lakarr2d, make_bdlknc_zones, make_bdlknc2d


class MFmodelMixin():
    """Base class for MFsetup models.
    """

    def __init__(self, parent):

        # property attributes
        self._nper = None
        self._perioddata = None
        self._sr = None
        self._modelgrid = None
        self._bbox = None
        self._parent = parent
        self._parent_layers = None
        self._parent_mask = None
        self._lakarr_2d = None
        self._isbc_2d = None
        self._ibound = None
        self._lakarr = None
        self._isbc = None
        self._lake_bathymetry = None
        self._precipitation = None
        self._evaporation = None
        self._lake_recharge = None

        # flopy settings
        self._mg_resync = False

        self._features = {}  # dictionary for caching shapefile datasets in memory

        # arrays remade during this session
        self.updated_arrays = set()

        # cache of interpolation weights to speed up regridding
        self._interp_weights = None

    @property
    def _lakarr2d(self):
        """2-D array of areal extent of lakes. Non-zero values
        correspond to lak package IDs."""
        if self._lakarr_2d is None:
            lakarr2d = np.zeros((self.nrow, self.ncol))
            if 'lak' in self.package_list:
                lakes_shapefile = self.cfg['lak'].get('source_data', {}).get('lakes_shapefile', {}).copy()
                lakesdata = self.load_features(**lakes_shapefile)  # caches loaded features
                lakes_shapefile['lakesdata'] = lakesdata
                lakes_shapefile.pop('filename')
                lakarr2d = make_lakarr2d(self.modelgrid, **lakes_shapefile)
            self._lakarr_2d = lakarr2d
            self._isbc_2d = None
        return self._lakarr_2d

