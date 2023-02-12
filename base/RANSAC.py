import numpy as np
from shapely.geometry.polygon import LineString
from base.Distances import lineseg_dists
from scipy import stats


class RansacIsochrone:
    def __init__(self, shift_x=0.1, shift_y=-0.75):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def create_multiline(self, points, shift_length=1e3):
        # Compute unit vector in opposite direction of line shift
        vec = np.array([-self.shift_x, -self.shift_y])
        # normalize vector  to unit length
        vec /= np.linalg.norm(vec)
        # create line with points as start point and shifted points as end point
        line = np.stack([points, points + vec*shift_length], axis=1)
        return line

    def intersection(self, points, line):
        A = self.create_multiline(points)
        res = np.array([line.intersects(LineString(a)) for a in A], dtype=int)
        return res

    def is_inside_main_and_binaries(self, points, isochrone):
        # Compute binary line
        binaries = np.copy(isochrone)
        binaries[:, 0] += self.shift_x
        binaries[:, 1] += self.shift_y
        # Convert into shapely linestrings
        isoline = LineString(isochrone)
        binaryline = LineString(binaries)
        # Compute intersections
        nb_intersections = self.intersection(points, isoline) + self.intersection(points, binaryline)
        return nb_intersections == 1

    def closest_point_on_isochrone_and_binary(self, points, isochrone):
        # closest point on isochrone
        closest_points_sl = lineseg_dists(points, isochrone[1:], isochrone[:-1])
        # closest point on binary isochrone
        binaries = np.copy(isochrone)
        binaries[:, 0] += self.shift_x
        binaries[:, 1] += self.shift_y
        closest_points_bl = lineseg_dists(points, binaries[1:], binaries[:-1])
        # compute minimum between both
        dists_sl = np.linalg.norm(closest_points_sl - points, axis=1)
        dists_bl = np.linalg.norm(closest_points_bl - points, axis=1)
        closest_isochrone_arg = np.argmin(np.stack([dists_sl, dists_bl]), axis=0)
        cond = closest_isochrone_arg == 0
        closest_points = np.where(cond[:, None], closest_points_sl, closest_points_bl)
        return closest_points

    @staticmethod
    def n_sigma_distance(n_sigma):
        """Mahanobolis distance to n sigma conversion"""
        quantile = stats.norm.cdf(n_sigma) - stats.norm.cdf(-n_sigma)
        return np.sqrt(-2 * np.log(1 - quantile))

    def is_inside_Nsigma_radius(self, points, std_devs, isochrone, n_sigma=3):
        """Check if points are N standard deviations from isochrone
        Assumes diagonal covariance matrix
        """
        # Compute component wise differences
        x1_y1, x2_y2 = np.square(points - self.closest_point_on_isochrone_and_binary(points, isochrone)).T
        # Get standard deviations
        std_x1, std_y1 = std_devs.T
        # Underestimation correction
        uc = 2
        # Compute Mahalanobolis distance
        dists = np.sqrt((x1_y1 / (uc * std_x1**2)) + (x2_y2 / (uc * std_y1**2)))
        # dists = np.sqrt((x1_y1 / std_x1) + (x2_y2 / std_y1))
        # Compute N sigma radius
        is_still_inside = dists < self.n_sigma_distance(n_sigma)
        return is_still_inside

    def fit(self, isochrone, points, std_devs, n_sigma=3):
        # check if points are inside main isochrone and binary isochrone
        is_inside = self.is_inside_main_and_binaries(points, isochrone)
        # check if points are inside N sigma radius
        is_still_inside = self.is_inside_Nsigma_radius(points, std_devs, isochrone, n_sigma=n_sigma)
        # combine both conditions
        is_still_inside = is_still_inside | is_inside
        return is_still_inside
