import numpy as np
import pandas as pd
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy import units as u
from astropy.coordinates import SkyCoord


class ExtinctionPrior:
    def __init__(self, fname_edenhofer2023):
        self.edh_dustmap = Edenhofer2023Query(
            map_fname=fname_edenhofer2023,
            load_samples=False,
            integrated=True,
            flavor='main',
            seed=None
        )

    def check_input(self, coord_input):
        if isinstance(coord_input, (pd.DataFrame, pd.Series)):
            return coord_input.values
        elif isinstance(coord_input, (list, tuple)):
            return np.array(coord_input)
        elif isinstance(coord_input, np.ndarray):
            return coord_input
        else:
            raise ValueError('Input should be a DataFrame, Series, list, or numpy array')

    def compute_prior(self, ra, dec, plx):
        ra = self.check_input(ra)
        dec = self.check_input(dec)
        plx = self.check_input(plx)
        # Build skycoord object
        c = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            distance=1000/plx * u.pc,
            frame='icrs'
        )
        E = self.edh_dustmap.query(c)
        A_V = E * 2.8
        return A_V
