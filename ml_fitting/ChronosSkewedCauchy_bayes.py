import numpy as np
from ml_fitting.ChronosBase import ChronosBase
from utils.utils import isin_range
import scipy.stats as st
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee


class ChronosSkewCauchyBayes(ChronosBase):
    def __init__(self, data, isochrone_files_base_path, file_ending, **kwargs):
        # Initialize super class
        super().__init__(data, isochrone_files_base_path, file_ending, **kwargs)
        # Update optimize function
        self.optimize_function = self.log_likelihood

    def auto_bounds(self):
        """Update auto_bounds function"""
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        # Set ranges on binary parameters
        skewness_range = (0.2, 0.8)
        scale_range = (0.025, 0.12)
        return logAge_range, feh_range, av_range, skewness_range, scale_range

    def set_bounds(self, logAge_range=None, feh_range=None, av_range=None,
                   skewness_range=None, scale_range=None):
        """Update set_bounds function"""
        if logAge_range is None:
            logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        if feh_range is None:
            feh_range = (
                np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity)
            )
        if av_range is None:
            av_range = (0, 3)
        # Set ranges on skewness and scale parameters
        if skewness_range is None:
            skewness_range = (0.2, 0.8)
        if scale_range is None:
            scale_range = (0.025, 0.12)
        # Set bounds
        self.bounds = (logAge_range, feh_range, av_range, skewness_range, scale_range)
        return

    def log_likelihood(self, theta):
        logAge, feh, A_V, skewness, scale = theta
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
        ll = self.compute_log_likelihood(
            x_data=dist_total[keep2fit],
            # weights=weights[keep2fit],
            skewness=skewness,
            scale=scale
        )
        return ll

    def compute_log_likelihood(self, x_data, skewness=0.5, scale=0.04):
        # Define distributions
        # skewness...skewness parameter; value determined via fit to Nuria's data. Default value is 0.5
        # scale...scale parameter; value determined via fit to Nuria's data. Default value is 0.04
        rv = st.skewcauchy(a=skewness, loc=0, scale=scale)
        # Compute log likelihood
        ll_skewcauchy = np.log(rv.pdf(x_data))
        # multiply by weights
        # ll = -np.sum(ll_skewcauchy + np.log(weights))
        ll = np.sum(ll_skewcauchy)
        return ll

    def log_prior(self, theta):
        # Uniform priors
        logAge, feh, A_V, skewness, scale = theta
        logAge_range, feh_range, av_range, skewness_range, scale_range = self.bounds
        if isin_range(logAge, *logAge_range) and isin_range(feh, *feh_range) and isin_range(A_V, *av_range) and \
                isin_range(skewness, *skewness_range) and isin_range(scale, *scale_range):
            return 0.0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def fit_bayesian(self, nwalkers=20, nsteps=1000, burnin=100):
        # Set initial positions
        ndim = len(self.bounds)
        pos = np.zeros((nwalkers, ndim))
        for i in range(ndim):
            pos[:, i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1], nwalkers)
        # Set sampler
        with Pool(processes=cpu_count()) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=pool)
            # Run burn-in
            pos, prob, state = sampler.run_mcmc(pos, burnin)
            sampler.reset()
            # Run MCMC
            sampler.run_mcmc(pos, nsteps)
        # Save samples
        samples = sampler.get_chain(flat=True)
        # Save best fit
        best_fit = samples[np.argmax(sampler.get_log_prob(flat=True))]
        return sampler, best_fit, samples
