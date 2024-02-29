import imf
import numpy as np


class MassFitter:
    def __init__(self, observed_masses, massfunc='kroupa', completeness_range=(0.05, 10), n_bins=20, n_draws=500,
                 mass_range=(10, 5_000)):
        self.observed_masses = observed_masses
        self.massfunc = massfunc
        self.completeness_range = completeness_range
        self.n_bins = n_bins
        self.bins = np.logspace(np.log10(completeness_range[0]), np.log10(completeness_range[1]), n_bins)
        # Number of draws to smooth the mass distribution
        self.n_draws = n_draws
        self.mass_range = mass_range

    def compute_observed_hist(self):
        hist, edges = np.histogram(self.observed_masses, bins=self.bins)
        return hist

    def set_observed_masses(self, observed_masses):
        self.observed_masses = observed_masses

    def set_bins(self, completeness_range, n_bins):
        self.completeness_range = completeness_range
        self.n_bins = n_bins
        self.bins = np.logspace(np.log10(completeness_range[0]), np.log10(completeness_range[1]), n_bins)

    def set_draws(self, n_draws):
        self.n_draws = n_draws

    def model_masses(self, cluster_mass):
        masses = []
        edges = None
        for i in range(self.n_draws):
            m_samples = imf.make_cluster(cluster_mass, massfunc=self.massfunc, verbose=False, silent=True)
            hist, edges = np.histogram(m_samples, bins=self.bins)
            masses.append(hist)
        hist = np.mean(masses, axis=0)
        return hist, edges

    def chi2_cluster_mass(self, cluster_mass, observed_masses_binned):
        if isinstance(cluster_mass, np.ndarray):
            cluster_mass = cluster_mass[0]
        hist, _ = self.model_masses(cluster_mass)
        return np.sum((observed_masses_binned - hist) ** 2 / hist)

    def grid_fitter(self, min_mass, max_mass, n_grid_pts, observed_masses_binned):
        """Implement grid fitter due to significant time savings"""
        mass_grid = np.linspace(min_mass, max_mass, n_grid_pts)
        chi2_res = [self.chi2_cluster_mass(mass_i, observed_masses_binned) for mass_i in mass_grid]
        return mass_grid[np.argmin(chi2_res)], np.diff(mass_grid)[0]

    def fit(self, n_iter=5, n_grid_pts=5):
        observed_masses_binned = self.compute_observed_hist()
        # chi2_vec = np.vectorize(self.chi2_cluster_mass)
        mass_lo, mass_hi = self.mass_range
        for _ in range(n_iter):
            best_mass_estimate, step_size = self.grid_fitter(
                min_mass=max((mass_lo, self.mass_range[0])),
                max_mass=min(mass_hi, self.mass_range[1]),
                n_grid_pts=n_grid_pts,
                observed_masses_binned=observed_masses_binned
            )
            mass_lo = best_mass_estimate - step_size
            mass_hi = best_mass_estimate + step_size
        return best_mass_estimate
