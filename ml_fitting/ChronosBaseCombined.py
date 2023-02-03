import numpy as np
from isochrone.PARSEC import PARSEC
from isochrone.Baraffe15 import Baraffe15
from ml_fitting.DistancesCombined import DistanceCombined
from utils.utils import isin_range
import imf
from skopt import gp_minimize


class ChronosBaseCombined:
    def __init__(self, data, models='parsec', **kwargs):
        # Set fitting kwargs
        self.fitting_kwargs = dict(
            fit_range=(-2, 10), do_mass_normalize=False, weights=None
        )
        # Check for Baraffe15 isochrones
        if ('baraffe' in models.lower()) or ('bhac' in models.lower()):
            self.isochrone_handler = Baraffe15('../data/baraffe_files/', file_ending='GAIA')
            self.fitting_kwargs['fit_range'] = (-2, 12)
        # Fail save is always PARSEC isochrones
        else:
            self.isochrone_handler = PARSEC('../data/parsec_files/', file_ending='dat')
        # Instantiate distance handler
        self.distance_handler = DistanceCombined(data=data, **kwargs)
        self.bounds = self.auto_bounds()
        self.kroupa_imf = imf.Kroupa()
        # Define optimization function
        self.optimize_function = None

    def update_data(self, data, **kwargs):
        self.distance_handler = DistanceCombined(data=data, **kwargs)

    def set_bounds(self, logAge_range=None, feh_range=None, av_range=None):
        if logAge_range is None:
            logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        if feh_range is None:
            feh_range = (
                np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity)
            )
        if av_range is None:
            av_range = (0, 3)
        self.bounds = (logAge_range, feh_range, av_range)

    def set_fitting_kwargs(self, **kwargs):
        for key in kwargs:
            self.fitting_kwargs[key] = kwargs[key]

    def auto_bounds(self):
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        return logAge_range, feh_range, av_range

    def bootstrap(self, bootstrap_frac: float, p: bool = True):
        """Wrapper function for easier"""
        self.distance_handler.bootstrap(bootstrap_frac, p)

    def keep_data(self, iso_coords):
        iso_range = np.min(iso_coords[:, 1]), np.max(iso_coords[:, 1])
        isin_magg_range = isin_range(self.distance_handler.fit_data['hrd'][:, 2], *self.fitting_kwargs['fit_range'])
        isin_iso_range = isin_range(self.distance_handler.fit_data['hrd'][:, 2], *iso_range)
        keep2fit = isin_magg_range & isin_iso_range
        return keep2fit

    def compute_fit_info(self, logAge, feh, A_V, g_rp, signed_distance=False):
        """Calculate the Mahanobolis distance to an isochrone"""
        # Get isochrone
        iso_coords = self.isochrone_handler.model(logAge, feh, A_V, g_rp=g_rp)
        # Get the distance to the isochrone
        near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords, g_rp=g_rp)
        distances_vec = self.distance_handler.cmd_data(g_rp=g_rp) - near_pt_on_isochrone
        # Minimize distance between photometric measurements and isochrones
        dist_color, dist_magg = distances_vec.T
        # # Divide by errors
        # if not signed_distance:
        #     # Don't want signed distance to be penalized
        #     dist_color /= self.distance_handler.fit_data['hrd_err'][:, 0]
        #     dist_magg /= self.distance_handler.fit_data['hrd_err'][:, 1]
        # Square values and add weight influence
        dist_total = np.sqrt(dist_color**2 + dist_magg**2)
        if signed_distance:
            dist_total *= np.sign(dist_color)
        # Compute mass
        masses = self.isochrone_handler.compute_mass(near_pt_on_isochrone, logAge, feh, A_V, g_rp)
        # compute points to fit
        keep2fit = self.keep_data(iso_coords)
        # Return the total distance
        return dist_total, masses, keep2fit

    def fit(self, **kwargs):
        # Define defaults
        n_calls = kwargs.pop('n_calls', 50)
        acq_func = kwargs.pop('acq_func', "gp_hedge")
        n_random_starts = kwargs.pop('n_random_starts', 5)
        noise = kwargs.pop('noise', 'gaussian')
        random_state = kwargs.pop('random_state', None)
        n_jobs = kwargs.pop('n_jobs', -1)
        res = gp_minimize(
            self.optimize_function,           # the function to minimize
            list(self.bounds),                # the bounds on each dimension of x
            acq_func=acq_func,                # the acquisition function
            n_calls=n_calls,                  # the number of evaluations of f
            n_random_starts=n_random_starts,  # the number of random initialization points
            noise=noise,                      # the noise level (default: gaussian)
            random_state=random_state,
            n_jobs=n_jobs
        )
        return res
