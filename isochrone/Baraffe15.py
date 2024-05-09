import re
import os
import glob
import copy
import numpy as np
import pandas as pd
from isochrone.ICBase import ICBase


class Baraffe15(ICBase):
    """Handling Gaia (E)DR3 photometric system"""
    def __init__(self, dir_path, file_ending='GAIA', nb_interpolated=400):
        super().__init__(nb_interpolated)
        # Save some PARSEC internal column names
        self.comment = r'!'
        self.colnames = {
            'mass': 'M/Ms',
            'logg': 'g',
            'teff': 'Teff',
            'age': 'logAge',
            'metal': 'feh',
            'gmag': 'G',
            'bp': 'G_BP',
            'rp': 'G_RP',
            'header_start': '! M/Ms',
            'age_start': '!  t (Gyr) ='
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
        print('Baraffe+15 isochrones read and processed!')
        return pd.concat(frames)

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        df_iso = pd.read_csv(fname, delim_whitespace=True, header=None, comment=self.comment)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '!' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        # Post process: add ages
        counter = 0
        nb_entries_per_isochrone = []
        logAge_info = []
        with open(fname) as f:
            line_info = []
            for i, line in enumerate(f):
                if line.startswith('!---'):
                    counter += 1
                    if counter == 2:
                        line_info.append(i + 1)
                    if counter == 3:
                        line_info.append(i - 1)

                if line.startswith(self.colnames['age_start']):
                    line = re.sub(r'\t', ' ', line)  # remove tabs
                    age = float(re.findall("\d+\.\d+", line)[0])
                    logAge = np.round(np.log10(age * 10**9), decimals=2)
                    logAge_info.append(logAge)
                    # Save infos
                    if len(line_info) > 0:
                        counter = 0
                        nb_entries_per_isochrone.append(line_info[1] - line_info[0])
                        line_info = []
        nb_entries_per_isochrone.append(line_info[1] - line_info[0])
        # --- Add age ---
        df_iso[self.colnames['age']] = -1.0
        rolling_sum = 0
        for entry, logAge in zip(nb_entries_per_isochrone, logAge_info):
            end = entry + rolling_sum + 1
            df_iso[self.colnames['age']].iloc[rolling_sum:end] = logAge
            rolling_sum = end
        # --- Add metal (at least 2 points to allow interpolation) ---
        # todo: fix interpolator to a single argument
        df_iso[self.colnames['metal']] = -0.1
        df_iso_1 = copy.deepcopy(df_iso)
        df_iso_1[self.colnames['metal']] = 0.1
        data = pd.concat([df_iso, df_iso_1])
        # --- Compute colors ---
        data[self.g_rp] = data[self.colnames['gmag']] - data[self.colnames['rp']]
        data[self.bp_rp] = data[self.colnames['bp']] - data[self.colnames['rp']]
        return data
