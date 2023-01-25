import numpy as np
from ml_fitting.ChronosBase import ChronosBase
import scipy.stats as st


# def loglikelihood(x_data, loc_diff, scale_single, scale_binaries, p_t):
def loglikelihood(x_data, weights, skewness=0.5, scale=0.04):
    # Define distributions
    # skewness...skewness parameter; value determined via fit to Nuria's data. Default value is 0.5
    # scale...scale parameter; value determined via fit to Nuria's data. Default value is 0.04
    rv = st.skewcauchy(a=skewness, loc=0, scale=scale)
    # Compute log likelihood
    ll_skewcauchy = np.log(rv.pdf(x_data))
    # multiply by weights
    # ll = -np.sum(ll_skewcauchy + np.log(weights))
    ll = -np.sum(ll_skewcauchy)
    return ll


class ChronosSkewCauchy(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Update fitting kwargs
        skewness = kwargs.pop('skewness', 0.5)
        scale = kwargs.pop('scale', 0.04)
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.isochrone_data_distances
        # set skewness
        self.fitting_kwargs['skewness'] = skewness
        self.fitting_kwargs['scale'] = scale

    def isochrone_data_distances(self, x):
        logAge, feh, A_V = x
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
            skewness=self.fitting_kwargs['skewness'],
            scale=self.fitting_kwargs['scale']
        )
        return ll
