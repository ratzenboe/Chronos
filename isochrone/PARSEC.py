import re
import os
import glob
import pandas as pd
from isochrone.ICBase import ICBase


class PARSEC(ICBase):
    """Handling Gaia (E)DR3 photometric system"""
    def __init__(self, dir_path, file_ending='dat', nb_interpolated=400):
        super().__init__(nb_interpolated)
        # Save some PARSEC internal column names
        self.comment = r'#'
        self.colnames = {
            'mass': 'Mass',
            'logg': 'logg',
            'teff': 'logTe',
            'age': 'logAge',
            'metal': 'MH',
            'gmag': 'Gmag',
            'bp': 'G_BPmag',
            'rp': 'G_RPmag',
            'header_start': '# Zini'
        }
        self.post_process = {self.colnames['teff']: lambda x: 10 ** x}
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
            # Post processing
            for col, func in self.post_process.items():
                df_iso[col] = df_iso[col].apply(func)
            frames.append(df_iso)
        print('PARSEC isochrones read and processed!')
        return pd.concat(frames)

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        df_iso = pd.read_csv(fname, delim_whitespace=True, comment=self.comment, header=None)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '#' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        # --- Compute colors ---
        df_iso[self.g_rp] = df_iso[self.colnames['gmag']] - df_iso[self.colnames['rp']]
        df_iso[self.bp_rp] = df_iso[self.colnames['bp']] - df_iso[self.colnames['rp']]
        return df_iso
