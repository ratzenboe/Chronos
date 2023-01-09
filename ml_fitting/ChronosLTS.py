import numpy as np
from ml_fitting.ChronosBase import ChronosBase


class ChronosLTS(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update fitting kwargs
        self.fitting_kwargs['frac_lts'] = kwargs.pop('frac_lts', 0.8)
        # Update optimize function
        self.optimize_function = self.isochrone_data_distances

    def isochrone_data_distances(self, x):
        logAge, feh, A_V = x
        iso_coords = self.isochrone_handler.model(logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp)
        # Compute distances to isochrone
        near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords)
        distances_vec = self.distance_handler.fit_data['hrd'] - near_pt_on_isochrone
        weights = np.ones(distances_vec.shape[0])
        if self.fitting_kwargs['do_mass_normalize']:
            # Compute masses from nearest points on isochrone
            masses = self.isochrone_handler.compute_mass(near_pt_on_isochrone, logAge, feh, A_V, g_rp=self.use_grp)
            # We multiply each distance by the inverse of its likelihood
            weights = 1 / self.kroupa_imf(masses)
            # weights = self.kroupa_imf(masses)
        if self.fitting_kwargs['weights'] is not None:
            weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]

        # Minimize distance between photometric measurements and isochrones
        dist_color, dist_magg = distances_vec.T
        # Divide by errors
        dist_color /= self.distance_handler.fit_data['hrd_err'][:, 0]
        dist_magg /= self.distance_handler.fit_data['hrd_err'][:, 1]
        # Square values and add weight influence
        dist_total = dist_color**2 + dist_magg**2
        # -- remove points outside range --
        keep2fit = self.keep_data(iso_coords)
        # Get final distances
        dist_total = dist_total[keep2fit]
        # LTS fitting: remove 1-frac_lts worst
        keep_n = int(np.sum(keep2fit) * self.fitting_kwargs['frac_lts'])
        argsorted_lts = np.argsort(dist_total)
        dists_weighted = dist_total * weights[keep2fit]
        dist_lts = dists_weighted[argsorted_lts][:keep_n]
        return np.sum(dist_lts)
