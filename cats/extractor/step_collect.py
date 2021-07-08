from os.path import join, basename
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import StdDevUncertainty
from astropy import units as u

from ..spectrum import Spectrum1D, SpectrumArray, SpectrumArrayIO, SpectrumList
from .steps import Step


class CollectObservationsStep(Step, SpectrumArrayIO):
    filename = "spectra.flex"

    def run(self, observatory, star, planet):
        files_fname = join(self.raw_dir, "*.fits")
        files = glob(files_fname)
        additional_data_fname = join(self.raw_dir, "*.csv")
        try:
            additional_data = glob(additional_data_fname)[0]
            additional_data = pd.read_csv(additional_data)
        except:
            additional_data = None

        speclist = []
        for f in tqdm(files):
            i = int(basename(f)[-8:-5])
            hdu = fits.open(f)
            wave = hdu[1].data << u.AA
            flux = hdu[2].data << u.one

            if additional_data is not None:
                add = additional_data.iloc[i]
                time = Time(add["time"], format="jd")
                airmass = add["airmass"]
                rv = add["barycentric velocity (Paranal)"] << (u.km / u.s)

            spectra = []
            orders = list(range(wave.shape[1]))
            for order in orders:
                for det in [1, 2, 3]:
                    w = wave[det - 1, order]
                    f = flux[det - 1, order]
                    if np.all(np.isnan(w)) or np.all(np.isnan(f)):
                        continue

                    # We just assume shot noise, no read out noise etc
                    unc = np.sqrt(np.abs(f))
                    unc = StdDevUncertainty(unc)
                    spec = Spectrum1D(
                        flux=f,
                        spectral_axis=w,
                        uncertainty=unc,
                        source="CRIRES+ Data Challenge 1",
                        star=star,
                        planet=planet,
                        observatory_location=observatory,
                        datetime=time,
                        reference_frame="telescope",
                        radial_velocity=rv,
                        airmass=airmass,
                    )
                    spectra += [spec]

            speclist += [SpectrumList.from_spectra(spectra)]
            hdu.close()

        times = [spec.datetime for spec in speclist]
        sort = np.argsort(times)
        speclist = [speclist[i] for i in sort]
        times = [times[i] for i in sort]

        data = SpectrumArray(speclist)
        self.save(data, self.savefilename)
        return data
