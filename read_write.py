"""
Load various kinds of data from disk
"""
import os.path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import yaml
try:
    from yaml import CLoader as Loader  # , CDumper as Dumper
except ImportError:
    print('LibYaml not installed, ')
    from yaml import Loader  # , Dumper


class read_write:
    """ wrapper class for IO handling """

    def __init__(self, config_file='config.yaml'):
        self.config = self.load_config(config_file)

        # exctract important parameters from config
        self.target = self.config['name_target']
        self.data_dir = os.path.join(
            self.config['path_exoSpectro'], self.target)
        self.input_dir = os.path.join(self.data_dir, self.config['dir_input'])
        self.output_dir = os.path.join(
            self.data_dir, self.config['dir_output'])

        self.intermediary_file = os.path.join(
            self.output_dir, self.config['file_intermediary'])
        self.renew_all = self.config['renew_all']

    def __load_yaml__(self, fname):
        """ load json data from file with given filename """
        with open(fname, 'r') as fp:
            return yaml.load(fp, Loader=Loader)
        raise IOError

    def load_config(self, filename='config.yaml'):
        """ Load configuration from file """
        filename = os.path.join(os.getcwd(), filename)
        return self.__load_yaml__(filename)

    def load_parameters(self):
        """ Load orbital parameters """
        par_file = os.path.join(self.input_dir, self.config['file_parameters'])
        par = self.__load_yaml__(par_file)

        # Convert all parameters into km and seconds
        r_sun = 696000      # Radius Sun
        r_jup = 71350       # Radius Jupiter
        au = 149597871      # Astronomical Unit
        secs = 24 * 60 * 60  # Seconds in a day

        par['r_star'] = par['r_star'] * r_sun
        par['r_planet'] = par['r_planet'] * r_jup
        par['sma'] = par['sma'] * au
        par['period'] = par['period'] * secs
        par['duration'] = par['duration'] * secs

        # Derived values
        par['A_planet'] = np.pi * par['r_planet']**2
        par['A_star'] = np.pi * par['r_star']**2
        par['A_atm'] = np.pi * \
            (par['r_planet'] + par['h_atm'])**2 - par['A_planet']
        par['A_planet'] = par['A_planet'] / par['A_star']
        par['A_atm'] = par['A_atm'] / par['A_star']
        par['A_planet+atm'] = par['A_planet'] + par['A_atm']

        return par

    def load_input(self, wl_grid):
        """ load input spectrum """
        input_file = os.path.join(
            self.input_dir, self.config['file_atmosphere'])
        input_spectrum = pd.read_table(
            input_file, header=None, delim_whitespace=True).values.swapaxes(0, 1)

        return self.interpolation(input_spectrum[0], input_spectrum[1], wl_grid)

    def load_observation(self, n_exposures):
        """ Load observation spectrum """
        obs_file = os.path.join(
            self.input_dir, self.config['file_observation'])
        obs = pd.read_table(obs_file, header=None, delim_whitespace=True)
        wl_tmp = obs[0].values
        obs.drop([0, *range(n_exposures + 1, obs.shape[1])],
                 axis=1, inplace=True)
        obs = obs.values.swapaxes(0, 1)
        #obs = interp1d(wl_tmp, obs, kind=config['interpolation_method'], fill_value='extrapolate')(wl_grid)
        return obs, wl_tmp

    def load_tellurics(self, wl_grid, n_exposures, apply_interp):
        """ Load telluric spectrum """
        tell_file = os.path.join(self.input_dir, self.config['file_telluric'])
        tell = pd.read_table(tell_file, header=None, delim_whitespace=True)
        wl_tmp = tell[0]
        tell.drop([0, *range(n_exposures + 1, tell.shape[1])],
                  axis=1, inplace=True)

        tell = tell.values.swapaxes(0, 1)
        if apply_interp:
            tell = interp1d(
                wl_tmp, tell, kind=self.config['interpolation_method'], fill_value='extrapolate')(wl_grid)
            return tell
        else:
            return wl_tmp.values, tell

    def interpolation(self, wl_old, spec, wl_new):
        """ interpolate spec onto wl_grid """
        return interp1d(wl_old, spec, kind=self.config['interpolation_method'], fill_value='extrapolate')(wl_new)

    def load_star_model(self, wl_grid, fwhm, width, apply_normal=True, apply_broadening=True, apply_interp=True):
        """ Load stellar model data and apply normalization and wavelength interploation """
        # Prepare file names
        star_flux_file = os.path.join(
            self.input_dir, self.config['file_star_flux'])
        star_intensities = self.config['star_intensities']
        star_data_file = {i: os.path.join(
            self.input_dir, self.config['file_star_data'].format(str(i))) for i in star_intensities}

        star_flux = pd.read_table(star_flux_file, header=None,
                                  delim_whitespace=True, usecols=(0, 1, 2)).values
        star_data = {i: pd.read_table(star_data_file[i], header=None, delim_whitespace=True, usecols=(0,
                                                                                                      1,)).values for i in star_intensities}

        # fix wavelenghts
        star_flux[:, 0] = star_flux[:, 0] * 0.1  # convert to nm
        for i in star_intensities:
            star_data[i][:, 0] *= 0.1

        # Interpolate to wavelength grid
        if apply_normal:
            normalization = self.interpolation(
                star_flux[:, 0], star_flux[:, 1] / star_flux[:, 2], wl_grid)
        if apply_interp:
            star_flux = self.interpolation(
                star_flux[:, 0], star_flux[:, 1], wl_grid)
        else:
            star_flux = star_flux[:, 1]

        if apply_broadening:
            star_flux = self.instrument_profile(star_flux, fwhm, width)

        if apply_interp:
            tmp = {i: self.interpolation(
                star_data[i][:, 0], star_data[i][:, 1], wl_grid) for i in star_intensities}
        else:
            tmp = {i: star_data[i][:, 1] for i in star_intensities}
        for i in star_intensities:
            if apply_normal:
                tmp[i] *= normalization
            if apply_broadening:
                tmp[i] = self.instrument_profile(tmp[i], fwhm, width)

        tmp[0.0] = np.zeros(len(wl_grid), dtype=float)
        star_data = pd.DataFrame.from_dict(tmp)
        #star_flux = pd.DataFrame(star_flux)

        return star_flux, star_data

    # Only apply instrumental profile to artificial spectra
    def instrument_profile(self, spectrum, fwhm, width):
        """ apply instrumental profile broadening to the spectrum """
        height = 0.08
        # x = -width ... +width
        x = np.arange(-width, width +
                      1, step=1, dtype=np.float)
        # Gaussian
        y = height * np.exp(-0.5 * (x * 2.67 / fwhm)**2)

        extspec = np.zeros(len(spectrum) + 2 * width, dtype=float)
        extspec[:width] = spectrum[0]
        extspec[width:-width] = spectrum
        extspec[-width:] = spectrum[-1]

        outspec = np.zeros(len(spectrum))
        for i in range(len(spectrum)):
            outspec[i] = np.sum(
                extspec[i:i + 2 * width + 1] * y)

        normali = np.sum(spectrum[width:-width]) / \
            np.sum(outspec[width:-width])
        outspec = outspec * normali
        return outspec

    def load_intermediary(self):
        """ load intermediary data products """
        intermediary_file = np.load(self.intermediary_file)
        F = intermediary_file['F']
        G = intermediary_file['G']
        wl_grid = intermediary_file['wl']
        tell = intermediary_file['tell']
        obs = intermediary_file['obs']
        return obs, tell, wl_grid, F, G

    def save_intermediary(self, obs, tell, wl_grid, F, G):
        """ save F and G for further use """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        np.savez(self.intermediary_file, F=F, G=G,
                 wl=wl_grid, tell=tell, obs=obs)
