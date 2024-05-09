import re
import numpy as np
import os
import glob
import pandas as pd
from isochrone.ICBase import ICBase


class Dartmouth(ICBase):
    """Handling Dartmouth Gaia DR2 photometric system"""
    def __init__(self, dir_path, file_ending='Gaia', nb_interpolated=400, verbose=False):
        super().__init__(nb_interpolated)
        # Dartmouth internal column names
        self.colnames = {
            'mass': 'M/Mo',
            'age': 'logAge',
            'metal': 'feh',
            'gmag': 'Gaia_G',
            'bp': 'Gaia_BP',
            'rp': 'Gaia_RP'
        }
        # Save data and rename columns
        self.verbose = verbose
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
        # df_tot = pd.concat(frames, ignore_index=True)
        df_tot = pd.concat(frames)
        # --- Compute colors ---
        df_tot[self.g_rp] = df_tot[self.colnames['gmag']] - df_tot[self.colnames['rp']]
        df_tot[self.bp_rp] = df_tot[self.colnames['bp']] - df_tot[self.colnames['rp']]
        print('Dartmouth isochrones read and processed!')
        return df_tot

    def extract_age(self, line):
        match = re.search(r"#AGE=\s*([0-9]+\.?[0-9]*)", line)
        if match:
            # Extract the first group (the number)
            return float(match.group(1))
        else:
            return None

    def read(self, file_path):
        # Initialize variables
        data_segments = []
        current_segment = []
        dataframes = []
        columns = [
            'EEP',
            self.colnames['mass'],
            'LogTeff',
            'LogG',
            'LogL/Lo',
            self.colnames['gmag'],
            self.colnames['bp'],
            self.colnames['rp'],
            self.colnames['age'],
            self.colnames['metal'],
        ]
        # Read the file
        with open(file_path, 'r') as file:
            for ith_line, line in enumerate(file):
                # Metallicity is stored in 4th line
                if ith_line == 3:
                    values = line.split()
                    feh_metallicity = float(values[5])  # sixth element (index 5)

                if line.startswith('#AGE='):
                    age = self.extract_age(line)
                    logAge = np.log10(age * 10 ** 6)
                    if self.verbose:
                        print(f'Processing age: {age} Myr (metallicity: [Fe/H] = {feh_metallicity})')
                    # If there is a current segment, add it to the data_segments list
                    if current_segment:
                        data_segments.append(current_segment)
                        current_segment = []
                elif not line.startswith('#'):
                    # Add data lines to the current segment
                    save_line = line.split()
                    save_line += [logAge, feh_metallicity]
                    if len(save_line) == 10:
                        current_segment.append(save_line)
        # Don't forget to add the last segment
        if current_segment:
            data_segments.append(current_segment)
            # Process each segment into a DataFrame
        for index, segment in enumerate(data_segments):
            df = pd.DataFrame(segment, columns=columns)
            df = df.astype(
                {
                    'EEP': int,
                    self.colnames['mass']: float,
                    'LogTeff': float,
                    'LogG': float,
                    'LogL/Lo': float,
                    self.colnames['gmag']: float,
                    self.colnames['bp']: float,
                    self.colnames['rp']: float,
                    self.colnames['age']: float,
                    self.colnames['metal']: float,
                }
            )
            dataframes.append(df)
        # return pd.concat(dataframes, ignore_index=True)
        return pd.concat(dataframes)
