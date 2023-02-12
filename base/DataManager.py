import numpy as np


class PhotManager:
    """Handling Gaia (E)DR3 photometric system"""
    def __init__(self, data,
                 mag_g_name='phot_g_mean_mag', mag_bp_name='phot_bp_mean_mag', mag_rp_name='phot_rp_mean_mag',
                 flux_g_name='phot_g_mean_flux', flux_g_error_name='phot_g_mean_flux_error',
                 flux_bp_name='phot_bp_mean_flux', flux_bp_error_name='phot_bp_mean_flux_error',
                 flux_rp_name='phot_rp_mean_flux', flux_rp_error_name='phot_rp_mean_flux_error',
                 parallax_name='parallax', parallax_error_name='parallax_error'
                 ):
        # Save data and rename columns
        data_processed, compute_errors, columns2rename = self.input_handler(
            data,
            phot_g_mean_mag=mag_g_name, phot_bp_mean_mag=mag_bp_name, phot_rp_mean_mag=mag_rp_name,
            phot_g_mean_flux=flux_g_name, phot_g_mean_flux_error=flux_g_error_name,
            phot_bp_mean_flux=flux_bp_name, phot_bp_mean_flux_error=flux_bp_error_name,
            phot_rp_mean_flux=flux_rp_name, phot_rp_mean_flux_error=flux_rp_error_name,
            parallax=parallax_name, parallax_error=parallax_error_name
        )
        self.data = data_processed.rename(columns=columns2rename)
        self.abs_Gmag = (self.data['phot_g_mean_mag'] + 5*np.log10(self.data['parallax']) - 10).values
        self.color_bprp = self.data['phot_bp_mean_mag'].values - self.data['phot_rp_mean_mag'].values
        self.color_grp = self.data['phot_g_mean_mag'].values - self.data['phot_rp_mean_mag'].values
        if compute_errors:
            # Compute uncertainties in HRD axes
            e_Gmag, e_GBPmag, e_GRPmag = self.compute_mag_errors()
            plx, e_plx = self.data['parallax'].values, self.data['parallax_error'].values
            self.e_abs_Gmag = np.sqrt(e_Gmag ** 2 + (5 / (np.log(10) * plx)) ** 2 * e_plx ** 2)
            self.e_color_bprp = np.sqrt(e_GBPmag ** 2 + e_GRPmag ** 2)
            self.e_color_grp = np.sqrt(e_Gmag ** 2 + e_GRPmag ** 2)
        else:
            self.e_abs_Gmag = self.e_color_bprp = self.e_color_grp = np.ones_like(self.abs_Gmag)

    def input_handler(self, data, **kwargs):
        all_cols_in_data = True
        column_map = {}
        for colname, target_name in kwargs.items():
            if colname in data.columns:
                column_map[target_name] = colname
            else:
                all_cols_in_data = False
        return data[list(column_map.keys())], all_cols_in_data, column_map

    def compute_mag_errors(self):
        """Compute magnitude errors from fluxes
        Zero point errors: see https://www.cosmos.esa.int/web/gaia/edr3-passbands
        """
        sigmaG_0 = 0.0027553202
        sigmaGBP_0 = 0.0027901700
        sigmaGRP_0 = 0.0037793818
        FG, e_FG = self.data['phot_g_mean_flux'].values, self.data['phot_g_mean_flux_error'].values
        FGBP, e_FGBP = self.data['phot_bp_mean_flux'].values, self.data['phot_bp_mean_flux_error'].values
        FGRP, e_FGRP = self.data['phot_rp_mean_flux'].values, self.data['phot_rp_mean_flux_error'].values
        e_Gmag = np.sqrt((-2.5 / np.log(10) * e_FG / FG) ** 2 + sigmaG_0 ** 2)
        e_GBPmag = np.sqrt((-2.5 / np.log(10) * e_FGBP / FGBP) ** 2 + sigmaGBP_0 ** 2)
        e_GRPmag = np.sqrt((-2.5 / np.log(10) * e_FGRP / FGRP) ** 2 + sigmaGRP_0 ** 2)
        return e_Gmag, e_GBPmag, e_GRPmag

    def flux2mag(self, flux, m_0):
        """Compute apparent magnitude from flux given zero point m_0"""
        return -2.5 * np.log10(flux) + m_0

    def fluxPlx2absMag(self, flux, plx, m_0):
        """Compute absolute magnitude from fluxes and parallax measurement given zero point m_0"""
        mag = -2.5 * np.log10(flux) + m_0
        abs_mag = mag + 5 * np.log10(plx) - 10
        return abs_mag

    def compute_mags_from_fluxes(self):
        """Compute magnitudes from fluxes
        Zero points: see https://www.cosmos.esa.int/web/gaia/edr3-passbands
        """
        # Zero points
        G_0 = 25.6873668671
        GBP_0 = 25.3385422158
        GRP_0 = 24.7478955012
        # Compute values
        mag_G = self.flux2mag(self.data['phot_g_mean_flux'].values, G_0)
        mag_GBP = self.flux2mag(self.data['phot_bp_mean_flux'].values, GBP_0)
        mag_GRP = self.flux2mag(self.data['phot_rp_mean_flux'].values, GRP_0)
        return mag_G, mag_GBP, mag_GRP
