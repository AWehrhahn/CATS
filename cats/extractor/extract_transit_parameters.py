import batman
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time
from scipy.optimize import curve_fit
from tqdm import tqdm


def model(x, theta):
    # x = time of observation
    # y = normalized light curve
    t = x
    t0, per, R_p, a, inc, ecc, w, u1, u2, lvl = theta

    params = batman.TransitParams()
    params.t0 = t0  # time of inferior conjunction
    params.per = per  # orbital period
    params.rp = R_p  # planet radius (in units of stellar radii)
    params.a = a  # semi-major axis (in units of stellar radii)
    params.inc = inc  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.u = [u1, u2]  # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # limb darkening model

    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    flux *= lvl

    return flux


def lnprior(theta, trange):
    t0, per, R_p, a, inc, ecc, w, u1, u2, lvl = theta
    tmin, tmax = trange
    # Planet size is positive and less than the star
    # planet orbit is outside the star (semi mayor axis greater than stellar radius)
    # inclination is defined for 0 to 180
    # periastron is defined for 0 to 360
    # elliptical orbit, eccentricity between 0 and 1
    if (
        0 < R_p < 1
        and 1 < a < np.abs(np.tan(np.radians(inc)))
        and 0 <= inc < 180
        and 0 <= w < 360
        and 0 <= ecc < 1
        and tmin < t0 < tmax
        and 0 < per < 365
        and 0 < lvl
    ):
        return 0
    else:
        return -np.inf


def lnlike(theta, x, y):
    m = model(x, theta)
    return -0.5 * np.sum((y - m) ** 2)


def lnprob(theta, x, y, trange):
    lp = lnprior(theta, trange)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)


def extract_transit_parameters(spectra, telluric, star, planet):
    names = ["t0", "Per", "$R_p$", "a", "inc", "ecc", "w", "u1", "u2", "cont"]

    # Sort the observations by date
    times = spectra.datetime
    sort = np.argsort(times)
    times = times[sort]

    # The telluric absorption influences the transmision spectrum
    # and needs to be accounted for
    norm = np.nanmean(telluric.flux[sort], axis=1)
    flux = np.nanmean(spectra.flux[sort] / norm[:, None], axis=1)
    flux /= np.nanpercentile(flux, 95)

    # Setup batman model
    # The first guess is based on the literature values of the planet
    t0 = times[len(times) // 2].mjd  # time of inferior conjunction
    per = planet.period.to_value("day")  # orbital period
    rp = (planet.radius / star.radius).to_value(
        1
    )  # planet radius (in units of stellar radii)
    a = (planet.sma / star.radius).to_value(
        1
    )  # semi-major axis (in units of stellar radii)
    inc = planet.inc.to_value("deg")  # orbital inclination (in degrees)
    ecc = planet.ecc.to_value(1)  # eccentricity
    w = planet.w.to_value("deg")  # longitude of periastron (in degrees)
    u1, u2 = 0.1, 0.3  # limb darkening coefficients [u1, u2, u3, u4]

    p0 = np.array([t0, per, rp, a, inc, ecc, w, u1, u2, 1.0])

    # plt.plot(times.mjd, flux)
    # plt.plot(times.mjd, model(times.mjd, p0))
    # plt.show()

    # Get a better guess using traditional least squares
    p0, _ = curve_fit(lambda x, *p: model(x, p), times.mjd, flux, p0=p0)

    # plt.plot(times.mjd, flux)
    # plt.plot(times.mjd, model(times.mjd, p0))
    # plt.show()

    # Finally run the MCMC
    ndim, nwalkers = len(p0), 300
    nsteps = 3000
    burn_in_length = 500

    tmin, tmax = times.mjd.min(), times.mjd.max()
    p0 = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lnprob,
        args=(times.mjd, flux, (tmin, tmax)),
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),],
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    # If we need to check the burn in:
    # samples = sampler.get_chain()
    # fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    # for i in range(ndim):
    #     axes[i].plot(samples[:, :, i], "k", alpha=0.3)
    #     axes[i].set_xlim(0, len(samples))
    #     axes[i].set_ylabel(names[i])

    # axes[-1].set_xlabel("step number")
    # plt.show()

    # And determine the results from the samples
    samples = sampler.get_chain(discard=burn_in_length, flat=True)
    p_mcmc = np.percentile(samples, [16, 50, 84], axis=0)
    p_mcmc[0], p_mcmc[1], p_mcmc[2] = (
        p_mcmc[1],
        p_mcmc[2] - p_mcmc[1],
        p_mcmc[1] - p_mcmc[0],
    )

    print("Parameter confidence regions:")
    for i in range(ndim):
        print(
            "  %s: %.2f + %.2f - %.2f"
            % (names[i], p_mcmc[0, i], p_mcmc[2, i], p_mcmc[1, i])
        )

    fig = corner.corner(samples, labels=names)
    plt.show()

    m = model(times.mjd, p_mcmc[0])
    plt.plot(times.mjd, flux, label="observation")
    plt.plot(times.mjd, m, label="model")
    plt.xlabel("Time [mjd]")
    plt.ylabel("Mean flux / Mean telluric")
    plt.show()

    # Save the results
    planet.t0 = Time(p_mcmc[0][0], format="mjd")
    planet.period = p_mcmc[0][1] * u.day
    planet.radius = p_mcmc[0][2] * star.radius
    planet.semi_major_amplitude = p_mcmc[0][3] * star.radius
    planet.inclination = p_mcmc[0][4] * u.deg
    planet.eccentricity = p_mcmc[0][5] * u.one
    planet.argument_of_periastron = p_mcmc[0][6] * u.deg

    return planet


if __name__ == "__main__":
    from os.path import dirname, join
    from ..data_modules.stellar_db import StellarDb
    from ..simulator.detector import Crires
    from ..spectrum import SpectrumArray

    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    detector = Crires("H/1/4", [1, 2, 3])
    star = Star.load(join(target_dir, "star.yaml"))
    spectra = SpectrumArray.read(join(target_dir, "spectra_normalized.npz"))

    planet = extract_transit_parameters(spectra, star)

    fname = join(target_dir, "planet.yaml")
    planet.save(fname)
