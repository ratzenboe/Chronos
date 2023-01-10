import numpy as np
from ml_fitting.ChronosBase import ChronosBase
from ml_fitting.RANSAC import RansacIsochrone


class ChronosRANSAC(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        shift_x = kwargs.get('shift_x', 0.1)
        shift_y = kwargs.get('shift_y', -0.75)
        self.fitter = RansacIsochrone(shift_x=shift_x, shift_y=shift_y)
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.count_inside

    def count_inside(self, x):
        """Computes MSAC score
        see https://de.wikipedia.org/wiki/RANSAC-Algorithmus#MSAC"""
        logAge, feh, A_V = x
        iso_coords = self.isochrone_handler.model(logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp)
        dist_total, masses, keep2fit = self.compute_fit_info(
            logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp, signed_distance=False
        )
        # Points to fit
        points = self.distance_handler.fit_data['hrd']
        errors = self.distance_handler.fit_data['hrd_err']
        points = points[keep2fit]
        errors = errors[keep2fit]
        # Compute number of points inside isochrone and binary line
        is_inside_lines = self.fitter.fit(iso_coords, points, errors, n_sigma=3)
        # Compute weights
        weights = np.ones(points.shape[0])
        if self.fitting_kwargs['do_mass_normalize']:
            # Compute masses from nearest points on isochrone
            near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords)
            masses = self.isochrone_handler.compute_mass(
                near_pt_on_isochrone[keep2fit], logAge, feh, A_V, g_rp=self.use_grp
            )
            # We multiply each distance by the inverse of its likelihood
            weights = 1 / self.kroupa_imf(masses)

        # Compute MSAC (M-Estimator SAmple Consensus) score
        # Score is the sum of the weights of the inliers
        inlier_weighted_distances = dist_total[keep2fit][is_inside_lines]
        limit = self.fitter.n_sigma_distance(4)  # Grow out to 4 sigma, then stop
        capped_distances = np.where(inlier_weighted_distances < limit, inlier_weighted_distances, limit)
        score = np.sum(capped_distances)    # * weights[is_inside_lines])  TODO: weights mess up fit, removed for now
        # Plus the sum of increased weights of the outliers. Error score is constant for outliers
        max_score = 10
        nb_outliers = np.sum(~is_inside_lines)
        score += nb_outliers * max_score
        return score
