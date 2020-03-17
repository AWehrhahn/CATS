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
from exoorbit import Orbit


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
    spectra = [
        spec / [np.nanpercentile(s.flux.to_value(u.one), 95) for s in spec]
        for spec in tqdm(spectra)
    ]

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


def create_stellar(wave, wrange, star, time, reference_frame):
    stellar = SmeStellar(star, linelist=f"{data_dir}/crires_h_1_4.lin", normalize=False)
    stellar = stellar.get(wrange, time)
    for s in stellar:
        s.meta["planet"] = planet
        s.meta["observatory_location"] = detector.observatory
    # stellar = stellar.shift(reference_frame)
    # stellar = stellar.resample(wave)
    return stellar


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


data_dir = join(dirname(__file__), "noise_zero")
target_dir = join(dirname(__file__), "extact_noise_zero")
files = join(data_dir, "b_1.fits")

linelist = f"{data_dir}/crires_h_1_4.lin"

detector = Crires("H/1/4", [1, 2, 3])
observatory = detector.observatory
wrange = detector.regions

# Load data from disk
# TODO load ALL files
spectra = [SpectrumList.read(f) for f in tqdm(glob(files)[:1])]
times = Time([spec[0].datetime for spec in tqdm(spectra)])

sort = np.argsort(times)
spectra = [spectra[i] for i in sort]
times = times[sort]

wave = [spec.wavelength for spec in spectra[0]]
wave_all = np.concatenate(wave).to_value(u.AA)

# Star and Planet nominal data
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]

# star = spectra[0][0].meta["star"]
# planet = star.planets["b"]
orbit = Orbit(star, planet)


observatory_location = detector.observatory
sky_location = star.coordinates
frame = TelescopeFrame(observatory_location, sky_location)

spectra = continuum_normalize(spectra, detector.blaze)

stellar = create_stellar(wave, wrange, star, times[0], spectra[0][0].reference_frame)
stellar = stellar.shift(frame)
stellar = stellar.resample(wave, method="linear")
stellar /= [np.nanmedian(spec) for spec in stellar.flux]


plt.plot(
    np.concatenate(stellar.wavelength).to_value(u.AA),
    np.concatenate(stellar.flux).value,
    "--",
    label="star",
)

plt.plot(
    np.concatenate(spectra[0].wavelength).to_value(u.AA),
    np.concatenate(spectra[0].flux).to_value(1),
    ":",
    label="observation",
)
plt.legend()
plt.show()
# TODO: account for airmass !!!
# divide by tellurics? at least to determine the planet transit!
# altaz = observer.altaz(times, target)
# airmass = altaz.secz.value
# spectra = [spectra[i] / telluric[i] for i in range(len(spectra))]

img = np.zeros((len(spectra), spectra[0].size))
for i, spec in tqdm(enumerate(spectra)):
    img[i] = np.concatenate([s.flux.to_value(u.one) for s in spec])

coeff = upper_envelope(wave_all, img[0])
envelope = np.polyval(coeff, wave_all)
plt.plot(wave_all, img[0])
plt.plot(wave_all, envelope)
plt.show()

pa = orbit.phase_angle(times)
idx = np.where(orbit.mu(times) > 0)[0][[0, -1]]

plt.plot(pa, np.nanmean(img, axis=1))
plt.vlines(pa[idx], 0.9, 1.1)
plt.show()

plt.imshow(img, aspect="auto", origin="lower", vmin=0, vmax=1)
plt.show()


# Prepare planet spectrum
planet_spectrum = PsgPlanetSpectrum(star, planet)
planet_spectrum.prepare(wrange)
planet_spectrum_input = [planet_spectrum.get(wrange, time) for time in times]

# Sort spectra by time
sort = np.argsort(times)

spectra = continuum_normalize(spectra, detector.blaze)

telluric = extract_telluric(spectra, star, detector.observatory)

stellar, star = extract_stellar_test(spectra, star, linelist)

intensities = extract_intensities(wrange, times, star, planet, linelist)

# Solve equation
area_planet = planet.area / star.area
area_atmosphere = np.pi * (planet.radius + planet.atm_scale_height) ** 2
area_atmosphere /= star.area
area_planet = area_planet.to_value(u.one)
area_atmosphere = area_atmosphere.to_value(u.one)


def standardize_spectrum(spectrum, wave, reference_frame="planet"):
    spec = spectrum.shift(reference_frame).resample(wave)
    spec = [spec.flux.to_value(u.one) for spec in spec]
    spec = np.array(spec)
    return spec


n = len(spectra)
f, g, w = [_ for _ in range(n)], [_ for _ in range(n)], [_ for _ in range(n)]
for i in range(len(spectra)):
    wave = [spec.wavelength for spec in spectra[i]]
    spec = [spec.flux.to_value(u.one) for spec in spectra[i]]
    spec = np.array(spec)

    rf_planet = PlanetFrame(Time.now(), star, planet)

    stel = standardize_spectrum(stellar[i], wave, rf_planet)
    inti = standardize_spectrum(intensities[i], wave, rf_planet)
    tell = standardize_spectrum(telluric[i], wave, rf_planet)

    f[i] = inti * tell * area_atmosphere
    g[i] = spec - (stel - inti * area_planet) * tell
    w[i] = [w.to_value(u.AA) for w in wave]

# Create common wavelength grid from all observations
# TODO: does that take to long?
wave = np.concatenate(w)
wave = np.unique(wave)


def nonlinear_leastsq(A, b, segment=5):
    from scipy.optimize import minimize
    from cats.solution import __difference_matrix__, best_lambda, Tikhonov

    def func(x, A, b):
        return A * x - b

    def reg(x, D):
        return D @ x

    def cost(x):
        cost = np.mean(func(x, A, b) ** 2)
        regul = regweight * np.mean(reg(x, D) ** 2)
        return cost + regul

    A = [a[segment] for a in A]
    b = [c[segment] for c in b]
    A = np.nan_to_num(np.asarray(A))
    b = np.nan_to_num(np.asarray(b))

    size = len(A[0])
    D = __difference_matrix__(size)
    regweight = best_lambda(np.mean(A, axis=0), np.mean(b, axis=0), plot=False)
    bounds = [(0, 1)] * size
    x0 = Tikhonov(np.mean(A, axis=0), np.mean(b, axis=0), regweight)
    x0 -= np.min(x0)
    x0 /= np.max(x0)

    res = minimize(
        cost,
        x0=x0,
        bounds=bounds,
        options={"maxiter": int(1e10), "maxfun": int(1e10), "iprint": 1},
    )
    return res.x


def gaussian_process(A, b, segment=5):
    import GPy

    X = [a[segment] for a in A]
    Y = [c[segment] for c in b]
    X = np.nan_to_num(np.asarray(X))
    Y = np.nan_to_num(np.asarray(Y))

    X = np.mean(X, axis=0)
    Y = np.mean(Y, axis=0)

    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=True)

    GPy.plotting.show(model.plot())

    return model


for seg in tqdm(range(24)):
    # planet_spectrum = gaussian_process(f, g)
    planet_spectrum = nonlinear_leastsq(f, g, seg)
    wave = spectra[0][seg].wavelength

    np.save(f"{target_dir}/planet_{seg}.npy", planet_spectrum)

    psi = planet_spectrum_input[0][seg].resample(wave)
    x = psi.wavelength
    y = psi.flux
    y = gaussian_filter1d(y, 5)

    plt.clf()
    plt.plot(x, y)
    plt.plot(wave, planet_spectrum)

    plt.savefig(f"{target_dir}/result_{seg}.png")
pass