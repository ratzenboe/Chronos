import numpy as np
from ml_fitting.DataManager import PhotManager
from utils.utils import chunks
import multiprocessing
from numba import jit
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


class Distance(PhotManager):
    """Class handling distance information between data and isochrones"""
    def __init__(self, data, use_grp=False, **kwargs):
        super().__init__(data, **kwargs)
        if use_grp:
            self.data_hrd = np.vstack([self.color_grp, self.abs_Gmag]).T
            self.data_e_hrd = np.vstack([self.e_color_grp, self.e_abs_Gmag]).T
        else:
            self.data_hrd = np.vstack([self.color_bprp, self.abs_Gmag]).T
            self.data_e_hrd = np.vstack([self.e_color_grp, self.e_abs_Gmag]).T
        # Remove nans
        self.is_not_nan = np.isnan(self.data_hrd).sum(axis=1) == 0
        self.data_hrd = self.data_hrd[self.is_not_nan]
        self.data_e_hrd = self.data_e_hrd[self.is_not_nan]
        # bootstrap data
        self.fit_data = {'hrd': copy.deepcopy(self.data_hrd), 'hrd_err': copy.deepcopy(self.data_e_hrd)}
        self.index_subset = np.arange(self.data_hrd.shape[0])

    def bootstrap(self, bootstrap_frac: float, p: bool = True):
        if isinstance(p, bool):
            if p:
                mg = self.data_hrd[:, 1]
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

    def nearest_points(self, iso_coords):
        return lineseg_dists(self.fit_data['hrd'], iso_coords[1:], iso_coords[:-1])

    def get_distances_chunk(self, isochrone_coords, ages, metals, avs, indices):
        """Calculate nearest distance to isochrone for data chunk"""
        distances_calculated = []
        for coord, idx in zip(isochrone_coords, indices):
            near_pt_on_isochrone = self.nearest_points(coord)
            distances_calculated.append(
                {
                    # 'color_dist': self.data_hrd[:, 0] - near_pt_on_isochrone[:, 0],
                    # 'mag_g_dist': self.data_hrd[:, 1] - near_pt_on_isochrone[:, 1],
                    'iso_data_dists': self.fit_data['hrd'] - near_pt_on_isochrone,
                    'near_pts': near_pt_on_isochrone,
                    'age': ages[idx],
                    'metal': metals[idx],
                    'a_v': avs[idx]
                }
            )
        return distances_calculated

    def distance_isochrone_data_multiple(self, iso_coords, ages, metals, avs):
        """Calculate the distance between a list of given isochrones and data points"""
        # 1) Calculate the distances between the data and the isochrones in a multiprocessing way
        # Multiprocessing: seperate lines into nb_processes & send each chunk to a processor
        nb_processes = min(multiprocessing.cpu_count(), len(iso_coords))
        pool = multiprocessing.Pool(processes=nb_processes)
        d_iso_lst = [
            pool.apply_async(self.get_distances_chunk,
                             (iso_coords_chunk, ages, metals, avs, idx_range))
            for iso_coords_chunk, idx_range in chunks(iso_coords, len(iso_coords) // nb_processes + 1)
        ]
        pool.close()
        pool.join()
        d_isos_full = [am_dist_dict for d_iso in d_iso_lst for am_dist_dict in d_iso.get()]
        return d_isos_full

    def distance_isochrone_data_single(self, iso_coords):
        near_pt_on_isochrone = self.nearest_points(iso_coords)
        distance_iso_pts = self.fit_data['hrd'] - near_pt_on_isochrone
        return distance_iso_pts

    def signed_distances_single(self, iso_coords):
        distance_iso_pts = self.distance_isochrone_data_single(iso_coords)
        return np.sign(distance_iso_pts[:, 0]) * np.linalg.norm(distance_iso_pts, axis=1)

    # @staticmethod
    # def pnt_dist(arr, line):
    #     return nearest_points(line, geom.Point(arr[0], arr[1]))[0].coords[0]
    #
    # def nearest_point_shapely(self, isochrone_coords):
    #     """Make precise calculations for points which do not fulfill the 2 neighbors criterium
    #     Use nearest_points function from geom -> Quite slow but only applied to a few points
    #     :param data_hrd: Numpy data w/ shape=(n_points, 2): Features color, mag
    #     :param isochrone_coords: Points sampled on isochrone; shape=(n_points, 2): Frist features bp_rp, second mag_g
    #     :return: Nearest point on isochrone (also between 2 points)
    #     """
    #     line = geom.LineString(isochrone_coords)
    #     near_pts = np.apply_along_axis(self.pnt_dist, 1, self.data_hrd, line)
    #     return near_pts