import numpy as np
from base.ChronosBase import ChronosBase


class ChronosLTS(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Update fitting kwargs
        frac_lts = kwargs.pop('frac_lts', 0.8)
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.isochrone_data_distances
        # set frac_lts
        self.fitting_kwargs['frac_lts'] = frac_lts

    def isochrone_data_distances(self, x):
        logAge, feh, A_V = x
        # Compute distances to isochrone
        dist_total, masses, keep2fit = self.compute_fit_info(
            logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp, signed_distance=False
        )
        # Compute weights
        weights = np.ones(dist_total.shape[0])
        # if self.fitting_kwargs['do_mass_normalize']:
            # We multiply each distance by the inverse of its likelihood
            # weights = 1 / self.kroupa_imf(masses)
            # weights = self.kroupa_imf(masses)
        if self.fitting_kwargs['weights'] is not None:
            weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]
        # Get final distances
        dist_total = dist_total[keep2fit]
        # LTS fitting: remove 1-frac_lts worst
        keep_n = int(np.sum(keep2fit) * self.fitting_kwargs['frac_lts'])
        argsorted_lts = np.argsort(dist_total)
        dists_weighted = dist_total**2 * weights[keep2fit]
        dist_lts = dists_weighted[argsorted_lts][:keep_n]
        return np.sum(dist_lts)
