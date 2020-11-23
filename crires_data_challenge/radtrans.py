import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from cats.spectrum import Spectrum1D

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc


def radtrans(wrange, star, planet):
    # 0.8 to 5
    wmin = wrange[0].to_value("um")
    wmax = wrange[1].to_value("um")
    # Initialize atmosphere
    # including the elements in the atmosphere
    atmosphere = Radtrans(
        line_species=["H2O", "H2", "CH4"],
        # line_species=["H2O", "CO_all_iso", "CH4", "CO2", "Na", "K"],
        rayleigh_species=["H2", "He"],
        continuum_opacities=["H2-H2", "H2-He"],
        wlen_bords_micron=[wmin, wmax],
        mode="lbl",
    )

    # Define planet parameters
    # Planet radius
    R_pl = planet.radius.to_value("cm")
    # R_pl = 1.838 * nc.r_jup_mean
    # surface gravity
    # gravity = 1e1 ** 2.45
    gravity = planet.surface_gravity.to_value("cm/s**2")
    # reference pressure (for the surface gravity and radius)
    # TODO: ????
    P0 = 0.01

    # Pressure in bar
    # has to be equispaced in log
    print("Setup atmosphere pressures")
    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)

    # Define temperature pressure profile
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 200.0
    # T_equ = 1500.0
    T_equ = planet.teff_from_stellar(star.teff).to_value("K")
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    # Define mass fractions
    mass_fractions = {}
    mass_fractions["H2"] = 0.74 * np.ones_like(temperature)
    mass_fractions["He"] = 0.24 * np.ones_like(temperature)
    mass_fractions["H2O"] = 0.001 * np.ones_like(temperature)
    mass_fractions["CH4"] = 0.000001 * np.ones_like(temperature)
    # mass_fractions["CO_all_iso"] = 0.01 * np.ones_like(temperature)
    # mass_fractions["CO2"] = 0.00001 * np.ones_like(temperature)
    # mass_fractions["Na"] = 0.00001 * np.ones_like(temperature)
    # mass_fractions["K"] = 0.000001 * np.ones_like(temperature)

    MMW = 2.33 * np.ones_like(temperature)

    # Calculate transmission spectrum
    print("Calculate transmission Spectrum")
    atmosphere.calc_transm(
        temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0
    )
    # atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW)
    wave = nc.c / atmosphere.freq / 1e-4
    flux = 1 - (atmosphere.transm_rad / nc.r_sun) ** 2

    wave = wave << u.um
    flux = flux << u.one

    spec = Spectrum1D(
        spectral_axis=wave,
        flux=flux,
        description="petitRADTRANS transmission spectrum",
        source="petitRADTRANS",
        reference_frame="barycentric",
        star=star,
        planet=planet,
    )

    return spec


# # Plot
# plt.rcParams["figure.figsize"] = (10, 6)
# rp = atmosphere.transm_rad
# rs = nc.r_sun
# flux = 1 - (rp / rs) ** 2
# plt.plot(nc.c / atmosphere.freq / 1e-4, flux)

# from hitran_linelist import HitranSpectrum

# hitspec = HitranSpectrum()
# plt.plot(hitspec.wavelength.to_value("um"), hitspec.flux, "--")

# plt.xscale("log")
# plt.xlabel("Wavelength (microns)")
# plt.ylabel(r"Transit radius ($\rm R_{Jup}$)")
# plt.show()
