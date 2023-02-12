import numpy as np
from base.DataManager import PhotManager
import copy
from scipy.stats import triang


# @jit(nopython=True)
def lineseg_dists(points, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    https://stackoverflow.com/questions/27161533

    Args:
        - points: np.array of shape (n, 2)
        - a: np.array of shape (m, 2)
        - b: np.array of shape (m, 2)
    """
    # normalized tangent vectors of line segments
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components
    # row-wise dot products of 2D vectors
    ap = np.transpose(np.expand_dims(a, axis=1) - points, (1, 0, 2))
    pb = np.transpose(points - np.expand_dims(b, axis=1), (1, 0, 2))
    s = np.sum(ap * d, axis=-1)
    t = np.sum(pb * d, axis=-1)
    # clamped parallel distance
    h = np.max([s, t, np.zeros_like(s)], axis=0)
    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = np.transpose(points - np.expand_dims(a, axis=1), (1, 0, 2))
    c = np.cross(d_pa, d)
    # Compute distances between data and line segments
    dists = np.hypot(h, c)
    # Get closest line segment
    is_closest = np.argmin(dists, axis=1)
    # Compute point on line segment that is closest to input points
    closest_points = a[is_closest] - d[is_closest] * np.expand_dims(s[np.arange(len(s)), is_closest], axis=1)
    return closest_points


class DistanceCombined(PhotManager):
    """Class handling distance information between data and isochrones"""
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.data_hrd = np.vstack([self.color_bprp, self.color_grp, self.abs_Gmag]).T
        self.data_e_hrd = np.vstack([self.e_color_bprp, self.e_color_grp, self.e_abs_Gmag]).T
        # Remove nans
        self.is_not_nan = np.isnan(self.data_hrd).sum(axis=1) == 0
        self.data_hrd = self.data_hrd[self.is_not_nan]
        self.data_e_hrd = self.data_e_hrd[self.is_not_nan]
        # bootstrap data
        self.fit_data = {'hrd': copy.deepcopy(self.data_hrd), 'hrd_err': copy.deepcopy(self.data_e_hrd)}
        self.index_subset = np.arange(self.data_hrd.shape[0])

    def cmd_data(self, g_rp=False):
        if g_rp:
            return self.fit_data['hrd'][:, [1, 2]]
        else:
            return self.fit_data['hrd'][:, [0, 2]]

    def cmd_errors(self, g_rp=False):
        if g_rp:
            return self.fit_data['hrd_err'][:, [1, 2]]
        else:
            return self.fit_data['hrd_err'][:, [0, 2]]

    def bootstrap(self, bootstrap_frac: float, p: bool = True):
        if isinstance(p, bool):
            if p:
                mg = self.data_hrd[:, 2]
                p = triang(c=0, loc=np.min(mg), scale=np.ptp(mg)).pdf(mg)
                p /= np.sum(p)
            else:
                p = None
        elif isinstance(p, np.ndarray):
            p /= np.sum(p)
        else:
            p = None
        self.index_subset = np.random.choice(
            self.is_not_nan.sum(), int(bootstrap_frac * self.is_not_nan.sum()),
            replace=True, p=p
        )
        self.fit_data['hrd'] = copy.deepcopy(self.data_hrd[self.index_subset])
        self.fit_data['hrd_err'] = copy.deepcopy(self.data_e_hrd[self.index_subset])
        return

    def nearest_points(self, iso_coords, g_rp=False):
        """Calculate nearest point on isochrone to data points"""
        if g_rp:
            return lineseg_dists(self.fit_data['hrd'][:, [0, 2]], iso_coords[1:], iso_coords[:-1])
        else:
            return lineseg_dists(self.fit_data['hrd'][:, [1, 2]], iso_coords[1:], iso_coords[:-1])
