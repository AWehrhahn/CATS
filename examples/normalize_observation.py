from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

import astroplan as ap
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import SME_Solver
from pysme.synthesize import synthesize_spectrum
from pysme.gui import plot_plotly
from tqdm import tqdm

from scipy.optimize import curve_fit

from cats.data_modules.sme import SmeIntensities, SmeStellar
from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.telluric_model import TelluricModel
from cats.data_modules.psg import PsgPlanetSpectrum
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList
from cats import reference_frame as rf
from cats.reference_frame import PlanetFrame, TelescopeFrame
from exoorbit import Orbit, Star


def round_to_nearest(value, options):
    value = np.atleast_2d(value).T
    options = np.asarray(options)

    diff = np.abs(value - options)
    sort = np.argsort(diff)
    nearest = options[sort[:, 0]]
    if value.size == 1:
        return nearest[0]
    return nearest


def continuum_normalize(spectra, blaze):
    # Correct for blaze function
    spectra = [spec / blaze for spec in tqdm(spectra)]

    # TODO Continuum normalize
    # Normalize to the same median
    # Without overlap between orders its going to be difficult to normalize
    # Maybe we can have some observations of the out of transit be in H/2/4 to fill the gaps?
    # We can't change it during transit, and the gaps are larger than the radial velocity shift
    for i, spec in tqdm(enumerate(spectra)):
        for j, s in enumerate(spec):
            f = s.flux.to_value(u.one)
            d = np.nanpercentile(f, 95)
            spectra[i][j]._data /= d

    return spectra


def continuum_normalize_part_2(spectra, stellar, telluric, detector):
    for j in tqdm(range(len(spectra))):
        spec = spectra[j]
        simulation = stellar[j] * telluric[j]
        simulation = detector.apply_instrumental_broadening(simulation)

        for i in tqdm(range(len(simulation))):
            x = spec[i].wavelength.to_value(u.AA)
            y = spec[i].flux.to_value(1)
            yp = simulation[i].flux.to_value(1)

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
            x, y, yp = x[mask], y[mask], yp[mask]
            if len(x) == 0:
                continue
            x0 = x[0]
            x -= x0

            def func(x, *c):
                return y * np.polyval(c, x)

            deg = 1
            p0 = np.ones(deg + 1)
            popt, pcov = curve_fit(func, x, yp, p0=p0)

            # For debugging
            # plt.plot(x, y * np.polyval(popt, x), label="observation")
            # plt.plot(x, yp, label="model")
            # plt.show()

            x = spec[i].wavelength.to_value(u.AA) - x0
            spectra[j][i]._data *= np.polyval(popt, x)

    return spectra


def extract_telluric(spectra, star, observatory):
    # Get telluric spectra (from observations?)
    # Shift everything to telescope restframe
    # Fit airmass versus spectrum, to get telluric

    # target_frame = rf.TelescopeFrame(Time.now(), observatory, star.coordinates)
    # spectra_telescope = [spec.shift(target_frame) for spec in tqdm(spectra)]
    # times = [spec[0].datetime for spec in tqdm(spectra_telescope)]

    # wave = [s.wavelength for s in spectra_telescope[0]]
    # for spec in spectra_telescope:
    #     for s in spec:
    #         s._data[np.isnan(s.flux)] = 1

    # spectra_telescope = [spec.resample(wave, method="spline") for spec in tqdm(spectra_telescope)]

    observer = ap.Observer(observatory)
    coords = star.coordinates
    target = ap.FixedTarget(name=star.name, coord=coords)
    altaz = observer.altaz(times, target)
    airmass = altaz.secz.value

    # npoints = spectra_telescope[0].shape[1]
    # sort = np.argsort(airmass)

    # nsegments = len(spectra_telescope[0])
    # coeff = [None for _ in range(nsegments)]
    # for i in range(nsegments):
    #     y = [spec[i].flux.to_value(u.one) for spec in spectra_telescope]
    #     y = np.array([np.nan_to_num(spec) for spec in y])
    #     coeff[i] = np.polyfit(airmass, y, 1)

    # # Compare to expectation
    # telluric_at_one = np.polyval(coeff[0], 1)

    model = TelluricModel(star, observatory)

    telluric = []
    for am, spec in zip(airmass, spectra):
        tell = model.interpolate_spectra(am)
        wrange = spec.regions
        telluric += [tell.extract_region(wrange)]

    return telluric


def extract_stellar_test(specta, star, linelist):
    # TODO: this just uses the nominal values from the star, instead of trying to fit
    stellar = SmeStellar(star, linelist=linelist, normalize=True)
    spec = stellar.get(spectra[0].regions, spectra[0][0].datetime)

    stellar = []
    for i in range(len(spectra)):
        spec.datetime = spectra[i][0].datetime
        stellar += [spec.shift("star")]

    return stellar, star


def extract_stellar(spectra, star, linelist):
    # Add all spectra together in barycentric frame
    # and then fit sme to get accurate stellar parameters
    spectra_star = [spec.shift("star") for spec in tqdm(spectra)]
    # spectra_star = spectra
    stellar = spectra_star[0]
    for spec in spectra_star:
        stellar += spec
    stellar /= len(spectra_star)

    # TODO: Set the mask?!!!!

    sme = SME_Structure()
    sme.wave = [s.wavelength.to_value(u.AA) for s in stellar]
    sme.spec = [s.flux.to_value(u.one) for s in stellar]

    sme.teff = star.teff.to_value(u.K)
    sme.logg = star.logg.to_value(u.one)
    sme.monh = star.monh.to_value(u.one)
    sme.vturb = star.vturb.to_value(u.km / u.s)

    vturb = star.vturb.to_value(u.km / u.s)
    vturb = round_to_nearest(vturb, [0, 1, 2])
    sme.atmo.source = f"marcs2012p_t{vturb:1.1f}.sav"
    sme.atmo.method = "grid"
    sme.atmo.geom = "PP"

    sme.abund = "solar"

    sme.linelist = ValdFile(linelist)

    sme.cscale_flag = "constant"
    # We determined the vrad value in a previous run
    # so we keep it fixed here to save time
    sme.vrad_flag = "fix"
    sme.vrad = -35.74602992

    solver = SME_Solver()
    sme = solver.solve(sme, ["teff", "logg", "monh"], segments=[5, 6])

    print("Old")
    print("Teff", star.teff)
    print("logg", star.logg)
    print("monh", star.monh)

    print("New")
    print("Teff", sme.teff)
    print("logg", sme.logg)
    print("monh", sme.monh)

    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename="stellar.html")

    # TODO Compare to nominal values for the star
    star.effective_temperature = sme.teff << u.K
    star.logg = sme.logg
    star.monh = sme.monh

    return sme.synth, star


def extract_intensities(wrange, times, star, planet, linelist):
    # Use the obs times to get phases and calculate specific intensities for those
    si = SmeIntensities(star, planet, linelist=linelist, normalize=True)
    si.prepare(wrange, times)
    intensities = [si.get(wrange, time) for time in times]
    return intensities


def extract_stellar_parameters(spectra, star, **kwargs):
    # TODO: actually fit the spectrum with SME
    return star


def create_stellar(wrange, star, planet, observatory, times, spectra):
    stellar = SmeStellar(star, linelist=f"{data_dir}/crires_h_1_4.lin", normalize=True)
    result = []
    for i, time in tqdm(enumerate(times)):
        reference_frame = spectra[i].reference_frame
        wave = spectra[i].wavelength
        spec = stellar.get(wrange, time)
        for s in spec:
            s.meta["planet"] = planet
            s.meta["observatory_location"] = observatory
        spec = spec.shift(reference_frame, inplace=True)
        spec = spec.resample(wave, method="linear")
        result += [spec]
    return result


def create_intensities(wrange, star, planet, observatory, times, spectra):
    stellar = SmeIntensities(
        star, planet, linelist=f"{data_dir}/crires_h_1_4.lin", normalize=True
    )
    stellar.prepare(wrange, times)
    result = []
    for i, time in tqdm(enumerate(times)):
        reference_frame = spectra[i].reference_frame
        wave = spectra[i].wavelength
        spec = stellar.get(wrange, time)
        for s in spec:
            s.meta["planet"] = planet
            s.meta["observatory_location"] = observatory
        spec = spec.shift(reference_frame, inplace=True)
        spec = spec.resample(wave, method="linear")
        result += [spec]
    return result


def create_telluric(wrange, star, planet, observatory, times, spectra):
    telluric = TelluricModel(star, observatory)
    result = []
    for i, time in tqdm(enumerate(times)):
        reference_frame = spectra[i].reference_frame
        wave = spectra[i].wavelength
        spec = telluric.get(wrange, time)
        for s in spec:
            s.meta["planet"] = planet
            s.meta["observatory_location"] = observatory
        spec = spec.shift(reference_frame, inplace=True)
        spec = spec.resample(wave, method="linear")
        result += [spec]
    return result


def upper_envelope(x, y, deg=5, factor=100):
    from scipy.optimize import minimize

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    offset_x = np.mean(x)
    offset_y = np.mean(y)
    x -= offset_x
    y -= offset_y

    def cost(p):
        m = np.polyval(p, x)
        c = m - y
        r = np.count_nonzero(c >= 0) / x.size
        return np.mean(c ** 2) + factor * (r - 0.95) ** 2

    x0 = np.polyfit(x, y, deg)
    res = minimize(cost, x0=x0, method="powell")
    coeff = res.x
    coeff[-1] += offset_y

    return res.x


data_dir = join(dirname(__file__), "noise_1")
target_dir = join(dirname(__file__), "noise_1_intermediate")
files = join(data_dir, "*.fits")

linelist = f"{data_dir}/crires_h_1_4.lin"

detector = Crires("H/1/4", [1, 2, 3])
observatory = detector.observatory
wrange = detector.regions

# Load data from disk
# TODO load ALL files
spectra = [SpectrumList.read(f) for f in tqdm(glob(files))]
times = Time([spec.datetime for spec in tqdm(spectra)])

sort = np.argsort(times)
spectra = [spectra[i] for i in sort]
times = times[sort]

wave = [spec.wavelength for spec in spectra[0]]
wave_all = np.concatenate(wave).to_value(u.AA)

# Star and Planet nominal data
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]
star = Star.load("star.yaml")
orbit = Orbit(star, planet)

spectra = continuum_normalize(spectra, detector.blaze)

# TODO: Extract stellar parameters from Spectra
stellar = create_stellar(wrange, star, planet, observatory, times, spectra)
# TODO: currently the tellurics is only based on the airmass at the time
# can we extract it from the observation somehow?
telluric = create_telluric(wrange, star, planet, observatory, times, spectra)

intensities = create_intensities(wrange, star, planet, observatory, times, spectra)

spectra = continuum_normalize_part_2(spectra, stellar, telluric, detector)

img_spectra = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(spectra)):
    img_spectra[i] = np.concatenate([s.flux.to_value(u.one) for s in spec])

img_telluric = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(telluric)):
    img_telluric[i] = np.concatenate([s.flux.to_value(u.one) for s in spec])

img_stellar = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(stellar)):
    img_stellar[i] = np.concatenate([s.flux.to_value(u.one) for s in spec])

img_intensities = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(intensities)):
    img_intensities[i] = np.concatenate([s.flux.to_value(u.one) for s in spec])


img_wave = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(spectra)):
    img_wave[i] = np.concatenate([s.wavelength.to_value(u.AA) for s in spec])


np.save("spectra.npy", img_spectra)
np.save("telluric.npy", img_telluric)
np.save("stellar.npy", img_stellar)
np.save("intensities.npy", img_intensities)

np.save("wave.npy", img_wave)
np.save("times.npy", [t.fits for t in times])

intermediate_dir = "noise_zero_intermediate"
intermediate_dir = join(dirname(__file__), intermediate_dir)
for i in range(len(spectra)):
    spectra[i].write(join(intermediate_dir, f"spectra_{i}.fits"))
    telluric[i].write(join(intermediate_dir, f"telluric_{i}.fits"))
    stellar[i].write(join(intermediate_dir, f"stellar_{i}.fits"))
    intensities[i].write(join(intermediate_dir, f"intensities_{i}.fits"))
