import numpy as np
from ml_fitting.ChronosBase import ChronosBase
import scipy.stats as st


# def loglikelihood(x_data, loc_diff, scale_single, scale_binaries, p_t):
def loglikelihood(x_data, weights):
    # Define distributions
    # a...skewness parameter; value determined via fit to Nuria's data
    # scale...scale parameter; value determined via fit to Nuria's data
    rv = st.skewcauchy(a=0.65, loc=0, scale=0.1)
    # Compute log likelihood
    ll_skewcauchy = np.log(rv.pdf(x_data))
    # multiply by weights
    # ll = -np.sum(ll_skewcauchy + np.log(weights))
    ll = -np.sum(ll_skewcauchy)
    return ll


class ChronosTdist(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.isochrone_data_distances

    def isochrone_data_distances(self, x):
        logAge, feh, A_V, loc_diff = x
        # Compute distances to isochrone
        dist_total, masses, keep2fit = self.compute_fit_info(
            logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp, signed_distance=True
        )
        # -- compute fitting weights --
        weights = np.ones_like(masses)
        if self.fitting_kwargs['do_mass_normalize']:
            # We multiply each distance by the inverse of its likelihood
            weights = 1 / self.kroupa_imf(masses)
            # weights = self.kroupa_imf(masses)
        if self.fitting_kwargs['weights'] is not None:
            weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]
        # Compute
        ll = loglikelihood(
            x_data=dist_total[keep2fit],
            weights=weights[keep2fit],
        )
        return ll
