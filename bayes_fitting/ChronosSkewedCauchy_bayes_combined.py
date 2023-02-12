import numpy as np
from base.ChronosBaseCombined import ChronosBaseCombined
from utils.utils import isin_range
import scipy.stats as st
import emcee


class ChronosSkewCauchyBayesCombined(ChronosBaseCombined):
    def __init__(self, data, **kwargs):
        # Initialize super class
        super().__init__(data, **kwargs)
        # Update optimize function
        self.optimize_function = self.log_likelihood

    def auto_bounds(self):
        """Update auto_bounds function"""
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        # Set ranges on binary parameters
        skewness_range_bprp = (0.2, 0.8)
        skewness_range_grp = (0.2, 0.8)
        scale_range_bprp = (0.01, 0.1)
        scale_range_grp = (0.0, 0.03)
        return logAge_range, feh_range, av_range, skewness_range_bprp, skewness_range_grp, scale_range_bprp, scale_range_grp

    def set_bounds(self, logAge_range=None, feh_range=None, av_range=None,
                   skewness_range_bprp=None, skewness_range_grp=None, scale_range_bprp=None, scale_range_grp=None):
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
        if skewness_range_bprp is None:
            skewness_range_bprp = (0.2, 0.8)
        if skewness_range_grp is None:
            skewness_range_grp = (0.2, 0.8)
        if scale_range_bprp is None:
            scale_range_bprp = (0.01, 0.1)
        if scale_range_grp is None:
            scale_range_grp = (0.0, 0.03)
        # Set bounds
        self.bounds = (logAge_range, feh_range, av_range,
                       skewness_range_bprp, skewness_range_grp, scale_range_bprp, scale_range_grp)
        return

    def log_likelihood(self, theta):
        logAge, feh, A_V, skewness_bprp, skewness_grp, scale_bprp, scale_grp = theta
        ll_tot = 0
        # for g_rp, skewness, scale in zip([False, True], [skewness_bprp, skewness_grp], [scale_bprp, scale_grp]):
        for g_rp, skewness, scale in zip([True, False], [skewness_grp, skewness_bprp], [scale_grp, scale_bprp]):
            # Compute distances to isochrone
            dist_total, masses, keep2fit = self.compute_fit_info(
                logAge=logAge, feh=feh, A_V=A_V, g_rp=g_rp, signed_distance=True
            )
            # # -- compute fitting weights --
            # weights = np.ones_like(masses)
            # if self.fitting_kwargs['do_mass_normalize']:
            #     # We multiply each distance by the inverse of its likelihood
            #     weights = 1 / self.kroupa_imf(masses)
            #     # weights = self.kroupa_imf(masses)
            # if self.fitting_kwargs['weights'] is not None:
            #     weights *= self.fitting_kwargs['weights'][self.distance_handler.is_not_nan]
            # Compute
            ll = self.compute_log_likelihood(
                x_data=dist_total[keep2fit],
                skewness=skewness,
                scale=scale
            )
            # ll_tot += ll  # /np.sum(keep2fit)
            ll_tot += ll  # ll_tot = g_rp
        return ll_tot

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
        logAge, feh, A_V, skewness_bprp, skewness_grp, scale_bprp, scale_grp = theta
        logAge_range, feh_range, av_range, skewness_range_bprp, skewness_range_grp, scale_range_bprp, scale_range_grp = self.bounds
        if isin_range(logAge, *logAge_range) and isin_range(feh, *feh_range) and isin_range(A_V, *av_range) and \
                isin_range(skewness_bprp, *skewness_range_bprp) and isin_range(skewness_grp, *skewness_range_grp) and \
                isin_range(scale_bprp, *scale_range_bprp) and isin_range(scale_grp, *scale_range_grp):
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
        # with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)  # pool=pool)
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
