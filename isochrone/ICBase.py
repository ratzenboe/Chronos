import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils.utils import resolve_duplicates

# ----- Corrective factors for extinction correction -----
corr_Gmag = 0.83627
corr_BPmag = 1.08337
corr_RPmag = 0.63439
corr_bprp = corr_BPmag - corr_RPmag
corr_grp = corr_Gmag - corr_RPmag
# --------------------------------------------------------


class ICBase:
    def __init__(self, nb_interpolated=400):
        self.abs_g = 'Gmag'
        self.bp_rp = 'color_BP_RP'
        self.g_rp = 'color_G_RP'
        # self.ccmd = ''
        self.nb_interpolation = nb_interpolated
        self.mass_grid = np.arange(nb_interpolated)
        self.unique_ages, self.age_mask = None, None
        self.unique_metallicity, self.metallicity_mask = None, None
        # RegularGridInterpolator
        self.rgi = {self.bp_rp: None, self.g_rp: None}
        # Approximate mass determination for each star via NearestNDInterpolator
        self.nndi = {self.bp_rp: None, self.g_rp: None}

    def check_inputs(self, logAge, feh, A_V):
        input_is_vector = False
        if isinstance(logAge, (np.ndarray, list, tuple)):
            min_age, max_age = np.min(logAge), np.max(logAge)
            min_feh, max_feh = np.min(feh), np.max(feh)
            min_av = np.min(A_V)
            input_is_vector = True
        else:
            min_age = max_age = logAge
            min_feh = max_feh = feh
            min_av = A_V
        # make checks
        is_ok_age = (min_age >= np.min(self.unique_ages)) & (max_age <= np.max(self.unique_ages))
        is_ok_metal = (min_feh >= np.min(self.unique_metallicity)) & (max_feh <= np.max(self.unique_metallicity))
        is_ok_av = min_av >= 0.
        # Raise errors
        if not is_ok_age:
            raise ValueError(f'Age {logAge} outside bounds ({np.min(self.unique_ages)}, {np.max(self.unique_ages)}).')
        if not is_ok_metal:
            raise ValueError(f'Metallicity {feh} outside bounds ({np.min(self.unique_metallicity)}, {np.max(self.unique_metallicity)}).')
        if not is_ok_av:
            raise ValueError(f'Extinction {A_V} outside bounds (0, inf).')
        return input_is_vector

    def fit(self, logAge, feh, A_V, color) -> np.ndarray:
        input_is_vector = self.check_inputs(logAge, feh, A_V)
        if input_is_vector:
            mass_grid_tiled = np.tile(self.mass_grid, (len(logAge), 1)).T
            # Swapaxes puts axis in front
            isochrone_coords = np.swapaxes(self.rgi[color]((feh, logAge, mass_grid_tiled)), 0, 1)
            # If the extinction array contains non-zero values
            if not np.any(A_V):
                mag_color, abs_mag_g = isochrone_coords.T
                extincted_color, extincted_magg = self.apply_extinction_by_color(abs_mag_g, mag_color, A_V, color)
                isochrone_coords = np.swapaxes(np.stack([extincted_color, extincted_magg]), 0, 2)
        else:
            isochrone_coords = self.rgi[color]((feh, logAge, self.mass_grid))
            if A_V > 0:
                mag_color, abs_mag_g = isochrone_coords.T
                extincted_color, extincted_magg = self.apply_extinction_by_color(abs_mag_g, mag_color, A_V, color)
                isochrone_coords = np.vstack([extincted_color, extincted_magg]).T
        return isochrone_coords

    def model(self, logAge: float, feh: float, A_V: float = 0, bp_rp: bool = True, g_rp: bool = False) -> np.ndarray:
        if g_rp:
            isochrone_coords = self.fit(logAge, feh, A_V, color=self.g_rp)
        elif bp_rp:
            isochrone_coords = self.fit(logAge, feh, A_V, color=self.bp_rp)
        else:
            raise NotImplementedError('Currently interpolation is implemented only in the 2D CMD. '
                                      'In the future we aim to expand interpolations to >=3 dimensions.')
        return isochrone_coords

    def model_pd(self, logAge: float, feh: float, A_V: float, bp_rp: bool = True, g_rp: bool = False) -> pd.DataFrame:
        isochrone_coords = self.model(logAge, feh, A_V, bp_rp, g_rp)
        if g_rp:
            return pd.DataFrame(isochrone_coords, columns=[self.g_rp, self.abs_g])
        elif bp_rp:
            return pd.DataFrame(isochrone_coords, columns=[self.bp_rp, self.abs_g])
        else:
            raise NotImplementedError('Currently interpolation is implemented only in the 2D CMD. '
                                      'In the future we aim to expand interpolations to >=3 dimensions.')

    def compute_mass(self, pt_on_iso_coords: np.ndarray,
                     logAge: float, feh: float, A_V: float, g_rp: bool = False) -> np.ndarray:
        # Set correct color
        color = self.g_rp if g_rp else self.bp_rp
        # First we need to compensate for extinction:
        # the nearest neighbors model does not account for extinction --> need to substract extinction (-A_V)
        mag_color, abs_mag_g = pt_on_iso_coords.T
        ext_c, ext_mg = self.apply_extinction_by_color(abs_mag_g, mag_color, -A_V, color)
        # Compute masses
        masses = self.nndi[color](
            np.full_like(ext_c, fill_value=logAge), np.full_like(ext_c, fill_value=feh), ext_c, ext_mg
        )
        return masses

    @staticmethod
    def interpolate_single_isochrone(color, abs_mag, mass, nb_interpolated):
        # Interpolated points along line:
        alpha = np.linspace(0, 1, nb_interpolated)
        has_potential_duplicates = True
        while has_potential_duplicates:
            points = np.vstack([color, abs_mag]).T
            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]
            try:
                interpolator = interp1d(distance, points, kind='slinear', axis=0)
                imass = interp1d(distance, mass, kind='slinear')
                has_potential_duplicates = False
            except ValueError:
                # If x or y has duplicates we perturb the data slightly
                color = resolve_duplicates(color)
                abs_mag = resolve_duplicates(abs_mag)
        # Interpolated points is 2d output
        interpolated_points = interpolator(alpha)
        # Interpolate mass along line
        mass_values = imass(alpha)
        return interpolated_points, mass_values

    def apply_extinction_by_color(self, abs_mag_g, mag_color, a_v, color):
        """Apply extinction to individual isochrone"""
        if isinstance(a_v, np.ndarray):
            av_add = a_v[None, :]
        elif isinstance(a_v, (list, tuple)):
            av_add = np.array(a_v)[None, :]
        else:
            av_add = a_v
        # Compute extincted magnitudes and colors
        extincted_magg = abs_mag_g + av_add * corr_Gmag
        if color == self.bp_rp:
            extincted_color = mag_color + av_add * corr_bprp
        else:
            extincted_color = mag_color + av_add * corr_grp
        return extincted_color, extincted_magg

    @staticmethod
    def distance_modulus(parallax):
        """Compute distance modulus (m-M)_0
        :param parallax: parallax in mas
        """
        return -5 * np.log10(parallax) + 10
