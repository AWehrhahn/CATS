import logging
from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
from specutils.spectra import SpectralRegion
from os.path import join, dirname

from astropy.time import Time
import astroplan as ap
from astroplan.plots import plot_airmass
from astropy.coordinates import SkyCoord

from cats.data_modules import psg, sme, stellar_db, telluric_model
from cats.simulator.simulator import Simulator
from cats.simulator.detector import Crires
from cats.simulator.noise import WhiteNoisePercentage

quantity_support()
# TODO: figure out logging (again)
logger = logging.getLogger(__name__)


def find_transit(observatory, star, planet):
    observer = ap.Observer(observatory)
    coords = star.coordinates
    target = ap.FixedTarget(name=star.name, coord=coords)
    system = ap.EclipsingSystem(
        primary_eclipse_time=planet.time_of_transit,
        orbital_period=planet.period,
        duration=planet.transit_duration,
        name=planet.name,
    )

    constraints = [
        ap.PrimaryEclipseConstraint(system),
        ap.AtNightConstraint.twilight_civil(),
        ap.AirmassConstraint(min=1, max=2),
    ]

    # TODO: Pick a good transit time automatically
    transit_time = system.next_primary_eclipse_time(Time.now(), 100)
    mask = ap.is_event_observable(constraints, observer, target, transit_time)
    transit_time = transit_time[mask[0]][1]

    plot_airmass(target, observer, transit_time)
    plt.vlines(transit_time.plot_date, 0, 3)
    plt.vlines(
        (transit_time - planet.transit_duration).plot_date, 0, 3, linestyles="dashed"
    )
    plt.vlines(
        (transit_time + planet.transit_duration).plot_date, 0, 3, linestyles="dashed"
    )
    plt.show()

    return transit_time


# TODO: get transit times with astroplan
# and compare to my internal calculations in ExoOrbit
data_directory = "/DATA/exoSpectro"
# data_directory = join(dirname(__file__), "../data")
target_directory = join(dirname(__file__), "noise_1")
detector = Crires("H/1/4", [1, 2, 3])

# Define wavelength range
wrange = detector.regions
blaze = detector.blaze

# Load orbital and stellar parameters from Database
sdb = stellar_db.StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]
observatory = detector.observatory

# Find next transit
# transit = find_transit(observatory, star, planet)
transit_time = "2020-05-25T10:31:25.418"
transit_time = Time(transit_time, format="fits")

planet.time_of_transit = transit_time

# Prepare stellar spectrum
stellar = sme.SmeStellar(star, linelist=f"{data_directory}/crires_h_1_4.lin")

# Prepare stellar intensities
intensities = sme.SmeIntensities(
    star, planet, linelist=f"{data_directory}/crires_h_1_4.lin"
)

# Prepare telluric spectrum
telluric = telluric_model.TelluricModel(star, observatory)

# Prepare planet spectrum
planet_spectrum = psg.PsgPlanetSpectrum(star, planet)

# Run Simulation
noise = []
noise = detector.load_noise_parameters()
noise += [WhiteNoisePercentage(0.01)]
sim = Simulator(
    detector, star, planet, stellar, intensities, telluric, planet_spectrum, noise=noise
)
spec = sim.simulate_series(wrange, transit_time, 68)

for i, s in enumerate(spec):
    s.write(f"{target_directory}/{planet.name}_{i}.fits", detector=detector)

# Compare to pure stellar spectrum
# Note: probably with different rest frame
# star_spec = stellar.get(wrange, 0)

planet_spectrum = planet_spectrum.get(wrange, transit_time)
planet_spectrum.write("planet_spectrum.fits")


for i in range(len(spec)):
    plt.clf()
    for s in spec[i]:
        plt.plot(s.wavelength, s.flux.decompose())
    plt.savefig(f"{target_directory}/plot.png")


pass
