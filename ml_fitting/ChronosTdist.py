import numpy as np
from ml_fitting.ChronosBase import ChronosBase
import scipy.stats as st


# def loglikelihood(x_data, loc_diff, scale_single, scale_binaries, p_t):
def loglikelihood(x_data, weights, loc_diff, scale_single, scale_binaries, p_t):
    # Define distributions
    rv_single = st.t(df=1, loc=0, scale=scale_single)
    rv_binaries = st.t(df=1, loc=loc_diff, scale=scale_binaries)
    # Compute log likelihood
    mix_model_ll = np.max([np.log(rv_single.pdf(x_data)*(1-p_t)), np.log(rv_binaries.pdf(x_data)*p_t)], axis=0)
    # mix_model_ll = np.max([np.log(rv_single.pdf(x_data)*p_t), np.log(rv_binaries.pdf(x_data)*(1-p_t))], axis=0)
    # multiply by weights
    # ll = -np.sum(mix_model_ll + np.log(weights))
    ll = -np.sum(mix_model_ll)
    return ll


def binary_fraction(mass, scale=1):
    """For now c_0 & c_1 are fixed. Model fit to data from Duchêne+2013"""
    c_0 = 0.38
    c_1 = 0.33
    bi_frac = scale * (c_0 * mass ** c_1)
    if isinstance(bi_frac, np.ndarray):
        bi_frac[bi_frac >= 0.95] = 0.95
    else:
        if bi_frac >= 0.95:
            bi_frac = 0.95
    return bi_frac


class ChronosTdist(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.isochrone_data_distances

    def set_bounds(self, logAge_range=None, feh_range=None, av_range=None,
                   loc_diff_range=None, scale_single_range=None, scale_binaries_range=None, scale_binscale_range=None):
        """Update set_bounds function"""
        if logAge_range is None:
            logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        if feh_range is None:
            feh_range = (
                np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity)
            )
        if av_range is None:
            av_range = (0, 3)
        # Set ranges on binary parameters
        if loc_diff_range is None:
            loc_diff_range = (0.05, 0.25)
        if scale_single_range is None:
            scale_single_range = (0.01, 0.3)
        if scale_binaries_range is None:
            scale_binaries_range = (0.01, 0.1)
        if scale_binscale_range is None:
            scale_binscale_range = (0.8, 1.1)
        # Set bounds variable
        self.bounds = (logAge_range, feh_range, av_range,
                       loc_diff_range, scale_single_range, scale_binaries_range, scale_binscale_range)
        return

    def auto_bounds(self):
        """Update auto_bounds function"""
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        # Set ranges on binary parameters
        loc_diff_range = (0.095, 0.105)
        scale_single_range = (0.01, 0.15)
        scale_binaries_range = (0.055, 0.07)
        scale_binscale_range = (0.8, 1.1)
        return logAge_range, feh_range, av_range, loc_diff_range, scale_single_range, scale_binaries_range, scale_binscale_range

    def isochrone_data_distances(self, x):
        logAge, feh, A_V, loc_diff, scale_single, scale_binaries, c_binary_func = x
        iso_coords = self.isochrone_handler.model(logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp)
        # Compute distances to isochrone
        near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords)
        distances_vec = self.distance_handler.fit_data['hrd'] - near_pt_on_isochrone
        # Compute masses from nearest points on isochrone
        masses = self.isochrone_handler.compute_mass(near_pt_on_isochrone, logAge, feh, A_V, g_rp=self.use_grp)
        # -- compute fitting weights --
        weights = np.ones_like(masses)
        if self.fitting_kwargs['do_mass_normalize']:
            # We multiply each distance by the inverse of its likelihood
            weights = 1 / self.kroupa_imf(masses)
            # weights = self.kroupa_imf(masses)
        if self.fitting_kwargs['weights'] is not None:
            weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]
        # weights = np.ones(distances_vec.shape[0])

        # --- Minimize distance between photometric measurements and isochrones ---
        dist_color, dist_magg = distances_vec.T
        # Divide by errors
        # dist_color /= self.distance_handler.data_e_hrd[:, 0]
        # dist_magg /= self.distance_handler.data_e_hrd[:, 1]
        # Square values and add weight influence
        dist_total = np.sqrt(dist_color**2 + dist_magg**2) * np.sign(dist_color)
        # Remove values outside range (use: M_G range)
        # -- remove points outside range --
        keep2fit = self.keep_data(iso_coords)
        # Compute
        ll = loglikelihood(
            x_data=dist_total[keep2fit],
            weights=weights[keep2fit],
            loc_diff=loc_diff,
            scale_single=scale_single,
            scale_binaries=scale_binaries,
            p_t=binary_fraction(masses[keep2fit], scale=c_binary_func)
        )
        return ll
