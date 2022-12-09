import numpy as np
import pandas as pd
from isochrone.PARSEC import PARSEC
from ml_fitting.Distances import Distance
from utils.utils import isin_range
import multiprocessing
import imf
from skopt import gp_minimize


class ChronosML:
    def __init__(self, data, isochrone_files_base_path, file_ending, use_grp=False, **kwargs):
        self.use_grp = use_grp
        self.isochrone_handler = PARSEC(isochrone_files_base_path, file_ending=file_ending)
        self.distance_handler = Distance(use_grp=use_grp, data=data, **kwargs)
        self.fitting_kwargs = dict(
            frac_lts=0.8, fit_range=(-2, 10), bootstrap_frac=0.8, do_mass_normalize=True, weights=None
        )
        self.bounds = self.auto_bounds()
        self.kroupa_imf = imf.Kroupa()

    def update_data(self, data, use_grp=None, **kwargs):
        if use_grp is None:
            use_grp = self.use_grp
        self.distance_handler = Distance(use_grp=use_grp, data=data, **kwargs)

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

    def set_fitting_kwargs(self,
                           frac_lts=0.8, fit_range=(-2, 10), bootstrap_frac=0.8, do_mass_normalize=True, weights=None):
        self.fitting_kwargs['frac_lts'] = frac_lts
        self.fitting_kwargs['fit_range'] = fit_range
        self.fitting_kwargs['bootstrap_frac'] = bootstrap_frac
        self.fitting_kwargs['do_mass_normalize'] = do_mass_normalize
        self.fitting_kwargs['weights'] = weights

    def auto_bounds(self):
        logAge_range = (np.min(self.isochrone_handler.unique_ages), np.max(self.isochrone_handler.unique_ages))
        feh_range = (np.min(self.isochrone_handler.unique_metallicity), np.max(self.isochrone_handler.unique_metallicity))
        av_range = (0, 3)
        return logAge_range, feh_range, av_range

    def minimize_lts(self, distances_vec, weights, seed=None):
        dist_color, dist_magg = distances_vec.T
        # Divide by errors
        dist_color /= self.distance_handler.data_e_hrd[:, 0]
        dist_magg /= self.distance_handler.data_e_hrd[:, 1]
        # Square values and add weight influence
        dist_total = dist_color**2 + dist_magg**2
        if isinstance(self.fitting_kwargs['bootstrap_frac'], (int, float, np.float)):
            np.random.seed(seed=seed)
            index_subset = np.random.choice(
                dist_total.size, int(self.fitting_kwargs['bootstrap_frac']*dist_total.size), replace=True
            )
            dist_total = dist_total[index_subset]
            weights = weights[index_subset]
            # Remove values outside of range
            isin_magg_range = isin_range(
                self.distance_handler.data_hrd[:, 1][index_subset],
                *self.fitting_kwargs['fit_range']
            )
        else:
            isin_magg_range = isin_range(self.distance_handler.data_hrd[:, 1], *self.fitting_kwargs['fit_range'])
        # Get final distances
        dist_total = dist_total[isin_magg_range]
        # LTS fitting: remove 1-frac_lts worst
        keep_n = int(np.sum(isin_magg_range) * self.fitting_kwargs['frac_lts'])
        argsorted_lts = np.argsort(dist_total)
        dists_weighted = dist_total * weights[isin_magg_range]
        dist_lts = dists_weighted[argsorted_lts][:keep_n]
        return dist_lts

    def isochrone_data_distances(self, x, seed=None):
        logAge, feh, A_V = x
        iso_coords = self.isochrone_handler.model(logAge=logAge, feh=feh, A_V=A_V, g_rp=self.use_grp)
        # Compute distances to isochrone
        near_pt_on_isochrone = self.distance_handler.nearest_points(iso_coords)
        distances_vec = self.distance_handler.data_hrd - near_pt_on_isochrone
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
        dist_lts = self.minimize_lts(distances_vec, weights, seed)
        return np.sum(dist_lts)

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
        # return np.array(res.x_iters)
        return res

    # def fit_bootstrap(self, logAge_0=7.5, feh_0=0., A_V_0=0., repetitions=20, method='trust-constr', **kwargs):
    #     if not isinstance(self.fitting_kwargs['bootstrap_frac'], (int, float, np.float)):
    #         self.fitting_kwargs['bootstrap_frac'] = 0.8
    #
    #     nb_processes = min(multiprocessing.cpu_count(), repetitions)
    #     pool = multiprocessing.Pool(processes=nb_processes)
    #     d_iso_lst = [
    #         pool.apply_async(
    #             self.fit,
    #             (logAge_0, feh_0, A_V_0, method, i),
    #             kwargs
    #         )
    #         for i in range(repetitions)
    #     ]
    #     pool.close()
    #     pool.join()
    #     fit_infos = np.array([best_fit_info.get() for best_fit_info in d_iso_lst])
    #     df_results = pd.DataFrame(fit_infos, columns=['logAge', 'feh', 'A_V'])
    #     return df_results

    # def check_grid_input(self, logAge_grid=None, feh_grid=None, A_V_grid=None):
    #     # -- check if inputs are given --
    #     if logAge_grid is None:
    #         logAge_grid = self.isochrone_handler.unique_ages.copy()
    #     if feh_grid is None:
    #         feh_grid = self.isochrone_handler.unique_metallicity.copy()
    #     if A_V_grid is None:
    #         A_V_grid = np.linspace(0, 1, 10)
    #     # -- int input --
    #     logAge_range, feh_range, av_range = self.bounds
    #     if isinstance(logAge_grid, int):
    #         logAge_grid = np.linspace(*logAge_range, logAge_grid)
    #     if isinstance(feh_grid, int):
    #         feh_grid = np.linspace(*feh_range, feh_grid)
    #     if isinstance(A_V_grid, int):
    #         A_V_grid = np.linspace(av_range, A_V_grid)
    #     return logAge_grid, feh_grid, A_V_grid
    #
    # def fit_grid(self, logAge_grid=None, feh_grid=None, A_V_grid=None, top_k=20):
    #     logAge_grid, feh_grid, A_V_grid = self.check_grid_input(logAge_grid, feh_grid, A_V_grid)
    #     print(f'Grid search on {logAge_grid.size*feh_grid*A_V_grid.size} solutions')
    #     xx, yy, zz = np.meshgrid(logAge_grid, feh_grid, A_V_grid)
    #     ages, metals, avs = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    #     # Get isochrones
    #     iso_coords = self.isochrone_handler.model(logAge=ages, feh=metals, A_V=avs, g_rp=self.use_grp)
    #     # Get distances between data and isochrones
    #     isodata_distances = self.distance_handler.distance_isochrone_data_multiple(iso_coords, ages, metals, avs)
    #     weights = np.ones(isodata_distances[0]['iso_data_dists'].shape[0])
    #     for iso_info in isodata_distances:
    #         if self.fitting_kwargs['do_mass_normalize']:
    #             # Compute masses from nearest points on isochrone
    #             masses = self.isochrone_handler.compute_mass(
    #                 iso_info['near_pts'], iso_info['age'], iso_info['metal'], iso_info['a_v'],
    #                 g_rp=self.use_grp
    #             )
    #             # We multiply each distance by the inverse of its likelihood
    #             weights = 1 / self.kroupa_imf(masses)
    #
    #         dlts = self.minimize_lts(iso_info['iso_data_dists'], weights=weights, seed=None)
    #         iso_info['sum_dists'] = np.sum(dlts)
    #     # Save best k models
    #     bestAges, bestFehs, bestAVs = [], [], []
    #     for i in np.argpartition([iso_info['sum_dists'] for iso_info in isodata_distances], top_k)[:top_k]:
    #         age, metal, av = isodata_distances[i]['age'], isodata_distances[i]['metal'], isodata_distances[i]['a_v']
    #         bestAges.append(age)
    #         bestFehs.append(metal)
    #         bestAVs.append(av)
    #
    #     return np.array(bestAges), np.array(bestFehs), np.array(bestAVs)

