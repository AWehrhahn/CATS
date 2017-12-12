import os.path
import numpy as np
import astropy.io.fits as fits

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


def load_keck_save(filename):
    """ just a reminder how to load the keck info file """
    import scipy.io
    import pandas as pd
    keck = scipy.io.readsav(filename)
    cat = keck['cat']
    df = pd.DataFrame.from_records(cat)
    df.applymap(lambda s: s.decode('ascii') if isinstance(s, bytes) else s)
    return df
