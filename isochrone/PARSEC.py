import re
import os
import glob
import numpy as np
import pandas as pd
from isochrone.ICBase import ICBase
from itertools import product
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


class PARSEC(ICBase):
    """Handling Gaia (E)DR3 photometric system"""
    def __init__(self, dir_path, file_ending='dat', nb_interpolated=400):
        super().__init__(nb_interpolated)
        # Save some PARSEC internal column names
        self.parsec_colnames = {
            'mass': 'Mass',
            'age': 'logAge',
            'metal': 'MH',
            'gmag': 'Gmag',
            'bp': 'G_BPmag',
            'rp': 'G_RPmag',
            'header_start': '# Zini'
        }
        # Save data and rename columns
        self.dir_path = dir_path
        self.flist_all = glob.glob(os.path.join(dir_path, f'*.{file_ending}'))
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.process_isochrone_infos()

    def read_files(self, flist):
        frames = []
        for fname in flist:
            df_iso = self.read(fname)
            frames.append(df_iso)
        return pd.concat(frames)

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        df_iso = pd.read_csv(fname, delim_whitespace=True, comment='#', header=None)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.parsec_colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(r'#', ' ', line)  # remove '#' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        # --- Compute colors ---
        df_iso[self.g_rp] = df_iso[self.parsec_colnames['gmag']] - df_iso[self.parsec_colnames['rp']]
        df_iso[self.bp_rp] = df_iso[self.parsec_colnames['bp']] - df_iso[self.parsec_colnames['rp']]
        return df_iso

    def remove_massive_end(self, df_iso):
        """Remove high mass end and mass decline (both cases can't be confidently fit to Gaia data)"""
        max_mass = 20
        mass = df_iso[self.parsec_colnames['mass']]
        max_index = np.argmax(mass > max_mass)        # Find first point with too high mass
        mass_loss_idx = np.argmax(np.diff(mass) < 0)  # Find mass decrease point
        stop_idx = min(mass_loss_idx, max_index)      # Take the first occurance of these 2 factors
        stop_idx = stop_idx if stop_idx > 0 else max(mass_loss_idx, max_index)
        if stop_idx == 0:
            stop_idx = -1
        return df_iso.iloc[:stop_idx]

    def built_interpolator_by_color(self, color):
        self.unique_ages, self.age_mask = np.unique(
            self.data[self.parsec_colnames['age']],
            return_inverse=True
        )
        self.unique_metallicity, self.metallicity_mask = np.unique(
            self.data[self.parsec_colnames['metal']],
            return_inverse=True
        )
        # Iterate first by second argument, then by first
        interpolated_isochrones = []
        mass_coordinates = []
        for metal_idx, age_idx in product(range(np.max(self.metallicity_mask) + 1), range(np.max(self.age_mask) + 1)):
            # Get entries in full dataframe
            entries_mask = (self.age_mask == age_idx) & (self.metallicity_mask == metal_idx)
            df_single_iso = self.remove_massive_end(self.data.loc[entries_mask])
            # Interpolate along single isochrones to have access to grid interpolation
            np_interpolation, mass_values = self.interpolate_single_isochrone(
                color=df_single_iso[color].values,
                abs_mag=df_single_iso[self.parsec_colnames['gmag']].values,
                mass=df_single_iso[self.parsec_colnames['mass']].values,
                nb_interpolated=self.nb_interpolation
            )
            interpolated_isochrones.append(np_interpolation)
            mass_coordinates.append(mass_values)
        # Concatenate interpolated 1D isochrones
        isos = np.concatenate(interpolated_isochrones)
        isos_reshaped = isos.reshape(self.unique_metallicity.size, self.unique_ages.size, self.nb_interpolation, 2)
        self.rgi[color] = RegularGridInterpolator(
            (self.unique_metallicity, self.unique_ages, np.arange(self.nb_interpolation)),
            isos_reshaped,
            method='linear'
        )
        # Create mass estimator (less precise than ages & metallicities as we don't have a regular grid
        # First age is varied, then metallicity
        ages = np.tile(self.unique_ages, (self.nb_interpolation, self.unique_metallicity.size)).T.ravel()
        metals = np.repeat(self.unique_metallicity, self.nb_interpolation*self.unique_ages.size)
        # Masses
        masses = np.concatenate(mass_coordinates)
        # print(ages.shape, metals.shape, isos.shape, isos[:, 0].shape, isos[:, 1].shape, masses.shape)
        age_metal_color_absmag = np.vstack([ages, metals, isos[:, 0], isos[:, 1]]).T
        self.nndi[color] = NearestNDInterpolator(x=age_metal_color_absmag, y=masses, rescale=False)
        return

    def process_isochrone_infos(self):
        """Process isochrone file and keep useful information"""
        for color in [self.bp_rp, self.g_rp]:
            self.built_interpolator_by_color(color)
        return
