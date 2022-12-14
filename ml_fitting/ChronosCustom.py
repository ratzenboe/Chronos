import numpy as np
from isochrone.PARSEC import PARSEC
from isochrone.Baraffe15 import Baraffe15
from ml_fitting.Distances import Distance
from utils.utils import isin_range
import imf
from skopt import gp_minimize
import scipy.stats as st


# def loglikelihood(x_data, weights, loc_diff, scale_single, scale_binaries, p_t):
def loglikelihood(x_data, loc_diff, scale_single, scale_binaries, p_t):
    # Define distributions
    rv_single = st.t(df=1, loc=0, scale=scale_single)
    rv_binaries = st.t(df=1, loc=loc_diff, scale=scale_binaries)
    # Compute log likelihood
    mix_model_ll = np.max([np.log(rv_single.pdf(x_data)*(1-p_t)), np.log(rv_binaries.pdf(x_data)*p_t)], axis=0)
    # mix_model_ll = np.max([np.log(rv_single.pdf(x_data)*p_t), np.log(rv_binaries.pdf(x_data)*(1-p_t))], axis=0)
    # multiply by weights
    # ll = -np.sum(mix_model_ll * np.log(weights))
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


class ChronosMixModel:
    def __init__(self, data, isochrone_files_base_path, file_ending, models='parsec', use_grp=False, **kwargs):
        self.use_grp = use_grp
        # Check for Baraffe15 isochrones
        if ('baraffe' in models.lower()) or ('bhac' in models.lower()):
            self.isochrone_handler = Baraffe15(isochrone_files_base_path, file_ending=file_ending)
        # Fail save is always PARSEC isochrones
        else:
            self.isochrone_handler = PARSEC(isochrone_files_base_path, file_ending=file_ending)
        self.distance_handler = Distance(use_grp=use_grp, data=data, **kwargs)
        self.fitting_kwargs = dict(fit_range=(-2, 10), do_mass_normalize=True, weights=None)
        self.bounds = self.auto_bounds()
        self.kroupa_imf = imf.Kroupa()

    def update_data(self, data, use_grp=None, **kwargs):
        if use_grp is None:
            use_grp = self.use_grp
        self.distance_handler = Distance(use_grp=use_grp, data=data, **kwargs)

    def set_bounds(self, logAge_range=None, feh_range=None, av_range=None,
                   loc_diff_range=None, scale_single_range=None, scale_binaries_range=None):
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
        # Set bounds variable
        self.bounds = (logAge_range, feh_range, av_range, loc_diff_range, scale_single_range, scale_binaries_range)
        return

    def set_fitting_kwargs(self, fit_range=(-2, 10), do_mass_normalize=True, weights=None):
        self.fitting_kwargs['fit_range'] = fit_range
        self.fitting_kwargs['do_mass_normalize'] = do_mass_normalize
        self.fitting_kwargs['weights'] = weights

    def auto_bounds(self):
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        # Set ranges on binary parameters
        # loc_diff_range = (0.05, 0.25)
        # scale_single_range = (0.01, 0.3)
        # scale_binaries_range = (0.01, 0.1)
        # -- updated ranges --
        loc_diff_range = (0.095, 0.105)
        scale_single_range = (0.01, 0.15)
        scale_binaries_range = (0.055, 0.07)
        return logAge_range, feh_range, av_range, loc_diff_range, scale_single_range, scale_binaries_range

    def keep_data(self, iso_coords):
        isin_magg_range = isin_range(self.distance_handler.fit_data['hrd'][:, 1], *self.fitting_kwargs['fit_range'])
        iso_range = np.min(iso_coords[:, 1]), np.max(iso_coords[:, 1])
        isin_iso_range = isin_range(self.distance_handler.fit_data['hrd'][:, 1], *iso_range)
        keep2fit = isin_magg_range & isin_iso_range
        return keep2fit

    def bootstrap(self, bootstrap_frac: float, p: bool = True):
        """Wrapper function for easier"""
        self.distance_handler.bootstrap(bootstrap_frac, p)

    # def determine_sign(self, masses, dist_color, dist_magg):
    #     sign_color = np.sign(dist_color)
    #     sign_magg = -np.sign(dist_magg)

    def isochrone_data_distances(self, x):
        logAge, feh, A_V, loc_diff, scale_single, scale_binaries = x
        iso_coords = self.isochrone_handler.model(logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp)
        # Compute distances to isochrone
        near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords)
        distances_vec = self.distance_handler.fit_data['hrd'] - near_pt_on_isochrone
        # Compute masses from nearest points on isochrone
        masses = self.isochrone_handler.compute_mass(near_pt_on_isochrone, logAge, feh, A_V, g_rp=self.use_grp)

        # if self.fitting_kwargs['do_mass_normalize']:
        #     # We multiply each distance by the inverse of its likelihood
        #     weights = 1 / self.kroupa_imf(masses)
        #     # weights = self.kroupa_imf(masses)
        # if self.fitting_kwargs['weights'] is not None:
        #     weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]
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
            # weights=weights[keep2fit],
            loc_diff=loc_diff,
            scale_single=scale_single,
            scale_binaries=scale_binaries,
            p_t=binary_fraction(masses[keep2fit])
        )
        return ll

    def fit(self, **kwargs):
        # Define defaults
        n_calls = kwargs.pop('n_calls', 50)
        acq_func = kwargs.pop('acq_func', "gp_hedge")
        n_random_starts = kwargs.pop('n_random_starts', 5)
        noise = kwargs.pop('noise', 'gaussian')
        random_state = kwargs.pop('random_state', None)
        n_jobs = kwargs.pop('n_jobs', -1)
        res = gp_minimize(
            self.isochrone_data_distances,    # the function to minimize
            list(self.bounds),                # the bounds on each dimension of x
            acq_func=acq_func,                # the acquisition function
            n_calls=n_calls,                  # the number of evaluations of f
            n_random_starts=n_random_starts,  # the number of random initialization points
            noise=noise,                      # the noise level (default: gaussian)
            random_state=random_state,
            n_jobs=n_jobs
        )
        return res
