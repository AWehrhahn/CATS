"""
Load various kinds of data from disk
"""
import os.path
import warnings
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.utils.exceptions import AstropyUserWarning
warnings.simplefilter('ignore', category=AstropyUserWarning)
import dateutil as du

import matplotlib.pyplot as plt

import yaml
try:
    from yaml import CLoader as Loader  # , CDumper as Dumper
except ImportError:
    print('LibYaml not installed, ')
    from yaml import Loader  # , Dumper


class read_write:
    """ wrapper class for IO handling """

    def __init__(self, config_file='config.yaml', dtype=np.float):
        self.config = self.load_config(config_file)
        self.dtype = dtype

        # exctract important parameters from config
        self.target = self.config['name_target']
        self.data_dir = os.path.join(
            self.config['path_exoSpectro'], self.target)
        self.input_dir = os.path.join(self.data_dir, self.config['dir_input'])
        self.output_dir = os.path.join(
            self.data_dir, self.config['dir_output'])
        self.intermediary_dir = os.path.join(
            self.data_dir, self.config['dir_intermediary'])

        self.config['intermediary_dir'] = self.intermediary_dir

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
        par['period_s'] = par['period'] * secs
        par['duration'] = par['duration'] * secs

        # Convert to radians
        par['inc'] = np.deg2rad(par['inc'])

        # Derived values, the pi factor gets canceled out
        par['A_planet'] = par['r_planet']**2
        par['A_star'] = par['r_star']**2
        par['A_atm'] = (par['r_planet'] + par['h_atm'])**2 - par['A_planet']
        par['A_planet'] = par['A_planet'] / par['A_star']
        par['A_atm'] = par['A_atm'] / par['A_star']
        par['A_planet+atm'] = par['A_planet'] + par['A_atm']

        # Overwrite standard values
        for k in par.keys():
            self.config[k] = par[k]

        return par

    def load_input(self, wl_grid):
        """ load input spectrum """
        input_file = os.path.join(
            self.input_dir, self.config['file_atmosphere'])
        input_spectrum = pd.read_table(
            input_file, header=None, delim_whitespace=True, dtype=self.dtype).values.swapaxes(0, 1)

        return self.interpolation(input_spectrum[0], input_spectrum[1], wl_grid)

    def load_observation(self, n_exposures='all'):
        """ Load observation spectrum """
        obs_file = os.path.join(
            self.input_dir, self.config['file_observation'])

        ext = os.path.splitext(obs_file)[1]
        if ext in ['.csv', '.dat']:
            obs = pd.read_table(obs_file, header=None,
                                delim_whitespace=True, dtype=self.dtype)
            wl_tmp = obs[0].values
            if n_exposures == 'all':
                n_exposures = obs.shape[1] - 1
            obs.drop([0, *range(n_exposures + 1, obs.shape[1])],
                     axis=1, inplace=True)
            obs = obs.values.swapaxes(0, 1)
            #obs = interp1d(wl_tmp, obs, kind=config['interpolation_method'], fill_value='extrapolate')(wl_grid)
            return wl_tmp, obs
        if ext in ['.ech', '.fits']:
            hdulist = fits.open(obs_file)
            header = hdulist[0].header
            tbdata = hdulist[1].data
            columns = hdulist[1].columns
            names = columns.names

            wave = tbdata[self.config['fits_wl']].reshape(-1)
            spec = tbdata[self.config['fits_flux']].reshape(-1)
            sig = tbdata[self.config['fits_sigma']].reshape(-1)
            if self.config['fits_cont'] is not None:
                cont = tbdata[self.config['fits_cont']].reshape(-1)
                spec /= cont

            sort = np.argsort(wave)
            wave = wave[sort]
            spec = spec[sort]
            sig = sig[sort]

            date = header[self.config['fits_date']]
            transit = self.config['transit']
            period = self.config['period']
            phase = ((date-transit)/period) % 1
            if phase > 0.5:
                phase = - (1-phase)
            phase = np.arcsin(phase)
            return wave, spec, [phase]

    def load_obs_xypoint(self):
        """ load an observation into a telfit xypoint structure """
        obs_file = os.path.join(
            self.input_dir, self.config['file_observation'])

        hdulist = fits.open(obs_file)
        header = hdulist[0].header
        data = hdulist[1].data
        orders = []
        for i in range(data['wave'].shape[1]):
            order = DataStructures.xypoint(x=data['wave'][:, i, :].reshape(-1),
                                           y=data['spec'][:, i, :].reshape(-1),
                                           cont=data['cont'][:,
                                                             i, :].reshape(-1),
                                           err=data['sig'][:, i, :].reshape(-1))
            orders.append(order)
        hdulist.close()
        return header, orders

    def load_tellurics_old(self, wl_grid, n_exposures, apply_interp):
        """ Load telluric spectrum """
        tell_file = os.path.join(
            self.input_dir, self.config['file_telluric'] + '.dat')
        ext = os.path.splitext(tell_file)[1]

        if ext in ['.dat', '.csv']:
            if ext == '.dat':
                tell = pd.read_table(tell_file, header=None,
                                     delim_whitespace=True, dtype=self.dtype)
            elif ext == '.csv':
                tell = pd.read_csv(tell_file, sep=',',
                                   header=None, dtype=self.dtype)
            wl_tmp = tell[0]
            tell.drop([0, *range(n_exposures + 1, tell.shape[1])],
                      axis=1, inplace=True)

            tell = tell.values.swapaxes(0, 1)
            if apply_interp:
                tell = np.interp(wl_grid, wl_tmp, tell)
                return tell
            else:
                return wl_tmp.values, tell

        if ext in ['.fits']:
            hdulist = fits.open(tell_file)
            tbdata = hdulist[1].data
            wl_tell = tbdata['lam']
            tell = tbdata['trans']
            if apply_interp:
                tell = np.interp(wl_grid, wl_tell, tell)
                return tell
            else:
                return wl_tell, tell

    def load_tellurics(self):
        tell_file = self.config['file_telluric'] + '_fit.fits'
        tell_file = os.path.join(self.config['intermediary_dir'], tell_file)
        tell = fits.open(tell_file)
        tbdata = tell[1].data
        wl = tbdata['mlambda'] * 1e4  # to Angstrom
        tell = tbdata['mtrans']
        return wl, tell

    def interpolation(self, wl_old, spec, wl_new):
        """ interpolate spec onto wl_grid """
        if not np.all(np.diff(wl_old) > 0):
            arg = np.argsort(wl_old)
            wl_old = wl_old[arg]
            spec = spec[arg]
        # interp1d(wl_old, spec, kind=self.config['interpolation_method'], fill_value='extrapolate')(wl_new)
        return np.interp(wl_new, wl_old, spec)

    def load_star_model(self, wl_grid, apply_normal=True, apply_interp=True):
        """ Load stellar model data and apply normalization and wavelength interploation """
        # Prepare file names
        star_flux_file = os.path.join(
            self.input_dir, self.config['file_star_flux'])
        star_intensities = self.config['star_intensities']
        del star_intensities[0]
        star_data_file = {i: os.path.join(
            self.input_dir, self.config['file_star_data'].format(str(i))) for i in star_intensities}

        star_flux = pd.read_table(star_flux_file, header=None,
                                  delim_whitespace=True, usecols=(0, 1, 2), dtype=self.dtype).values
        star_data = {i: pd.read_table(star_data_file[i], header=None, delim_whitespace=True, usecols=(
            0, 1,), dtype=self.dtype).values for i in star_intensities}

        # Interpolate to wavelength grid
        if apply_normal:
            normalization = self.interpolation(
                star_flux[:, 0], star_flux[:, 1] / star_flux[:, 2], wl_grid)

        if apply_interp:
            tmp = {i: self.interpolation(
                star_data[i][:, 0], star_data[i][:, 1], wl_grid) for i in star_intensities}
            tmp[0.0] = self.interpolation(
                star_flux[:, 0], star_flux[:, 2], wl_grid)
        else:
            tmp = {i: star_data[i][:, 1] for i in star_intensities}
            tmp[0.0] = np.copy(star_flux[:, 2])
        for i in tmp.keys():
            if apply_normal:
                tmp[i] *= normalization

        star_data = pd.DataFrame.from_dict(tmp)

        if apply_interp:
            if apply_normal:
                star_flux = self.interpolation(
                    star_flux[:, 0], star_flux[:, 1], wl_grid)
            else:
                star_flux = self.interpolation(
                    star_flux[:, 0], star_flux[:, 2], wl_grid)
        else:
            if apply_normal:
                star_flux = star_flux[:, 1]
            else:
                star_flux = star_flux[:, 2]

        return star_flux, star_data

    def load_marcs(self, wl_grid, apply_interp=True):
        """ load MARCS flux files """
        flux_file = os.path.join(
            self.input_dir, self.config['file_star_marcs'])
        wl_file = os.path.join(self.input_dir, self.config['file_star_wl'])
        flux = pd.read_table(flux_file, delim_whitespace=True,
                             header=None, names=['Flux']).values[:, 0]
        wl = pd.read_table(wl_file, delim_whitespace=True,
                           header=None, names=['WL']).values[:, 0]
        if apply_interp:
            flux = self.interpolation(wl, flux, wl_grid)

        star_int = {i: flux for i in self.config['star_intensities']}
        star_int = pd.DataFrame.from_dict(star_int)
        return flux, star_int

    # Only apply instrumental profile to artificial spectra
    def instrument_profile(self, spectrum, fwhm, width):
        """ apply instrumental profile broadening to the spectrum """
        height = 0.08
        # x = -width ... +width
        x = np.arange(-width, width +
                      1, step=1, dtype=self.dtype)
        # Gaussian
        y = height * np.exp(-0.5 * (x * 2.67 / fwhm)**2)

        extspec = np.zeros(len(spectrum) + 2 * width, dtype=self.dtype)
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

    def load_bin(self, filename='spectra0023IntFluxPlanetEarthContinum.bin'):
        """ load a binary data file """
        s = np.fromfile(os.path.join(self.input_dir, filename)
                        ).reshape((-1, 6)).swapaxes(0, 1)
        return s.astype(self.dtype)

    def convert_keck_fits(self):
        """ convert a keck file into something that MolecFit can use """
        hdulist = fits.open(os.path.join(
            self.input_dir, self.config['file_observation']))
        header = hdulist[0].header
        primary = fits.PrimaryHDU(header=header)
        wave = hdulist[1].data['WAVE']
        spec = hdulist[1].data['SPEC'] / hdulist[1].data['CONT']
        sig = hdulist[1].data['SIG'] / hdulist[1].data['CONT']

        wave = wave.reshape(-1)
        spec = spec.reshape(-1)
        sig = sig.reshape(-1)

        sort = np.argsort(wave)
        wave = wave[sort]
        spec = spec[sort]
        sig = sig[sort]

        col1 = fits.Column(name='WAVE', format='D', array=wave)
        col2 = fits.Column(name='SPEC', format='E', array=spec)
        col3 = fits.Column(name='SIG', format='E', array=sig)
        cols = fits.ColDefs([col1, col2, col3])
        tbhdu = fits.BinTableHDU.from_columns(cols)

        new = fits.HDUList([primary, tbhdu])
        new.writeto(os.path.join(self.intermediary_dir,
                                 self.config['file_observation_intermediary']), overwrite=True)

        # Update molecfit parameter file
        mfit = os.path.join(self.input_dir, self.config['file_molecfit'])
        with open(mfit, 'r') as f:
            p = f.readlines()

        def find(data, label):
            return [i for i, k in enumerate(data) if k.startswith(label)][0]

        index = find(p, 'filename:')
        p[index] = 'filename: ' + os.path.join(
            self.intermediary_dir, self.config['file_observation_intermediary']) + '\n'

        index = find(p, 'output_name:')
        p[index] = 'output_name: ' + self.config['file_telluric'] + '\n'

        index = find(p, 'rhum:')
        p[index] = 'rhum: ' + str(header['relhum'] * 100) + '\n'

        index = find(p, 'telalt:')
        p[index] = 'telalt: ' + str(abs(float(header['AZ']))) + '\n'

        index = find(p, 'utc:')
        utc = header['utc']
        utc = du.parser.parse(utc).time()
        utc = int(60 * 60 * utc.hour + 60 * utc.minute +
                  utc.second + 1e-3 * utc.microsecond)
        p[index] = 'utc: ' + str(utc) + '\n'

        mfit = os.path.join(self.intermediary_dir,
                            self.config['file_molecfit'])
        with open(mfit, 'w') as f:
            for item in p:
                f.write(item)
