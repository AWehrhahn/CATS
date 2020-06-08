# TODO
from glob import glob
from os.path import dirname, expanduser, join
from time import strftime

import astroplan
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits
from astropy import units as u

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray, SpectrumList
from cats.extractor.prepare import create_telluric

from selenite import load_store
from selenite.data_containers import shard
from selenite.fit_selenite import fit_selenite
from selenite.load_store import load_fits as load_fits
from selenite.load_store import load_normalized_spectra as load_norms
from selenite.load_store import read_db as read_db
from selenite.load_store import write_db as write_db
from selenite.load_store import write_fits as write_fits
from selenite.load_store import write_normalized_spectra as write_norms
from selenite.model import cluster_analysis as cluster_analysis
from selenite.model import fit_model as fit_model
from selenite.model import generate_model as generate_model
from selenite.model import generate_pwv_metric as generate_pwv_metric
from selenite.model import get_calibrators as get_calibrators
from selenite.model import regression_model as regression_model
from selenite.model import telluric_identification as telluric_id
from selenite.preprocessing import align_spectra as align_spectra
from selenite.preprocessing import constrain_wvs as constrain_wvs
from selenite.preprocessing import filter_bstars as filter_bstars
from selenite.preprocessing import normalize as normalize
from selenite.preprocessing import remove_fringes as remove_fringes
from selenite.preprocessing import suppress_stellar_lines as supression
from selenite.visualize import plot_data as plot_data
from selenite.visualize import plot_model as plot_model
from selenite.visualize import plot_PCCs as plot_PCCs
from selenite.visualize import plot_property as plot_property
from selenite.visualize import plot_regressions as plot_regressions
from selenite.visualize import plot_uncertainty as plot_uncertainty

# Unfortunately we had to copy paste a lot of the code from selenite in here, to handle
# the CRIRES+ format
# TODO: Seperate Selenite functions and put them back into selenite


def get_airmass(sl):
    time = sl.datetime
    observer = astroplan.Observer(sl.meta["observatory_location"])
    target = astroplan.FixedTarget(coord=sl.meta["star"].coordinates)
    altaz = observer.altaz(time, target)
    airmass = altaz.secz.value
    return airmass


def load_star(spectrum, shard_dict, blaze, orders):

    # 1: Open star's file
    # f = fits.open(fname, do_not_scale_image_data=True)
    # f = SpectrumList.read(fname)

    # 2: Load selected orders into shard_dict
    for order in orders:

        # 2(a): Get shard lin_x, lin_y, airmass, continuum and uncertainities
        lin_x = spectrum[order].wavelength.to_value("AA")
        lin_y = spectrum[order].flux.to_value(1)
        if blaze is not None:
            continuum = blaze[order].to_value(1)
        else:
            continuum = np.ones_like(lin_y)
        continuum *= np.nanpercentile(lin_x / continuum, 95)
        uncertainty = np.ones_like(lin_y) * continuum * 0.1
        z = float(get_airmass(spectrum))

        # 2(b): Strip nans
        good_px = np.logical_not(np.isnan(lin_y))
        print("good_px len", len(good_px))
        lin_x = lin_x[good_px]
        lin_y = lin_y[good_px]
        continuum = continuum[good_px]
        uncertainty = uncertainty[good_px]
        snrs = lin_y / uncertainty
        print("lin_y", lin_y)

        # 2(c): If a shard doesn't exist for the current order in shard_dict,
        #       create one.
        if order not in shard_dict:
            shard_dict[order] = shard.Shard(order)

        # 2(d): Set all lin_y data < 0.00001 to 0.00001, and then log lin_y.
        MIN_LIN_VAL = 0.00001  # 0.01
        try:
            lin_y[lin_y < MIN_LIN_VAL] = MIN_LIN_VAL
        except:
            print(lin_y)
        log_y = np.log(lin_y)

        # 2(e): Store results
        shard_dict[order].spectra[spectrum.datetime.fits] = shard.Spectrum_Data(
            lin_x, log_y, z, continuum, uncertainty, snrs
        )


def normalize_bstars(config_path, spectra, normed_bstar_dir):
    # 1) LOAD DATA
    # Loads: (a) config, (b) calibration spec., (c) a wavelength grid spec.
    # Stores each calibration spec. order in a data container called a shard.
    # Produces dictionary linking each shard to its order.
    np.seterr(all="raise")
    config = yaml.safe_load(open(config_path, "r"))
    orders = config["orders"]

    shard_dict = {}
    for spectrum in spectra:
        load_star(spectrum, shard_dict, blaze, orders)
    plot_data.plot_data(shard_dict, "wv", "log", config["plot_raw_data"])

    # 2) PREPROCESS B STARS
    # (i) Remove low SNR B stars,
    # (ii) Truncate B stars down to high SNR wavelength ranges.
    plot_uncertainty.plot_bstar_snrs(shard_dict, config, config["plot_bstar_snrs"])
    filter_bstars.filter_lo_snr_bstars(shard_dict, config)
    plot_uncertainty.plot_bstar_snrs(
        shard_dict, config, config["plot_bstar_snrs_after_filtering"]
    )

    remove_fringes.find_uncertainty_cutoffs(shard_dict, config)
    plot_uncertainty.plot_uncertainty_cutoffs(
        shard_dict, config["plot_bstar_uncertainty"]
    )

    plot_uncertainty.plot_bstar_cutoffs(shard_dict, config["plot_bstar_cutoffs"])
    remove_fringes.remove_fringes(shard_dict)
    plot_uncertainty.plot_bstar_cutoffs(shard_dict, config["plot_cut_bstar_cutoffs"])

    # 3) NORMALIZE B STARS
    normalize.normalize(shard_dict)
    plot_data.plot_data(shard_dict, "wv", "log", config["plot_norm_data"])

    # 4) SAVE RESULTS
    write_norms.write_normalized_spectra(shard_dict, normed_bstar_dir)
    return


def calibrate_selenite(wv_grid_path, db_path, config_path):

    # 1) LOAD DATA
    # Loads: (a) config, (b) calibration spec., (c) a wavelength grid spec.
    # Stores each calibration spec. order in a data container called a shard.
    # Produces dictionary linking each shard to its order.
    config = yaml.safe_load(open(config_path, "r"))
    normed_bstar_path = expanduser(config["normed_bstar_path"]).format(selenite=local)
    order = config["orders"]

    shard_dict = load_norms.load_normalized_bstars(normed_bstar_path, order)
    wv_grid_shard_dict = {}
    spectrum = SpectrumList.read(wv_grid_path)
    load_star(spectrum, wv_grid_shard_dict, blaze, order)

    # 2) FILTER BSTARS, SUPRESS LINES
    # Filter out all b stars with unflattened stellar lines/supress these lines
    filter_bstars.filter_bstars_with_unflatted_stellar_lines(shard_dict)
    supression.suppress_stellar_lines(shard_dict)
    plot_data.plot_data(
        shard_dict, "wv", "log", config["plot_filtered_data"], "after filtering"
    )

    # 3) CONSTRAIN WV_GRID TO GOOD WV RANGE
    constrain_wvs.constrain_wv_grid_wvs(shard_dict, wv_grid_shard_dict)

    # 4) ALIGN BSTARS TO WAVELENGTH GRID
    align_spectra.align_spectra(shard_dict, wv_grid_shard_dict)
    plot_data.plot_data(
        shard_dict, "wv", "log", config["plot_aligned_data"], "after alignment"
    )

    # 5) IDENTIFY TELLURIC PIXELS
    # i)   Get calibration line/calibration line suite data.
    # ii)  Identify pixels with significant PCC with either a) a H20 calibration line or b) z,
    #      or that are saturated.
    # iv)  Remove all 1 and 2 telluric pixel clusters.
    # v)   Remove all telluric clusters not in the shape of a Gaussian trough.
    # vi)  Remove all telluric clusters more than 1nm from another cluster.
    # vii)  Mark each cluster as non-water, water, or both.
    calibrator_px = telluric_id.find_calibrator_px(shard_dict, config)
    calibrators = telluric_id.generate_calibrators(shard_dict, calibrator_px)
    plot_property.plot_property(
        shard_dict,
        calibrators,
        "z",
        "wavelength",
        calibrator_px,
        config["plot_shard_zs"],
    )
    plot_property.plot_property(
        shard_dict,
        calibrators,
        "PWV_out",
        "wavelength",
        calibrator_px,
        config["plot_shard_PWVs"],
    )

    k = telluric_id.compute_PCC_threshold(config["p_value"], config["thresholds_file"])
    telluric_id.flag_high_PCC_pixels(calibrators, k, shard_dict, config)
    cluster_analysis.identify_clusters(shard_dict)
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "water", False, config["plot_water_PCCs"]
    )
    plot_PCCs.plot_coadd_spec_PCCs(shard_dict, "airmass", False, config["plot_z_PCCs"])
    plot_PCCs.plot_coadd_spec_PCCs(shard_dict, "water", True, config["plot_water_px"])
    plot_PCCs.plot_coadd_spec_PCCs(shard_dict, "airmass", True, config["plot_z_px"])

    # cluster_analysis.remove_1_and_2_pixel_clusters(shard_dict)
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "airmass", True, config["plot_z_px_no_fp"], "1"
    )
    # cluster_analysis.remove_non_trough_clusters(shard_dict, config)
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "airmass", True, config["plot_z_px_no_fp"], "2"
    )
    cluster_analysis.remove_isolated_clusters(shard_dict)
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "airmass", True, config["plot_z_px_no_fp"], "3"
    )

    # 6) EXPAND CLUSTERS
    # Expand each cluster by one pixel on either side to pick up pixels in its line's tail.
    cluster_analysis.expand_clusters(shard_dict, config)
    fp_ttl = "w/ fp removal (& expansion)"
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "water", True, config["plot_water_px_no_fp"], fp_ttl
    )
    plot_PCCs.plot_coadd_spec_PCCs(
        shard_dict, "airmass", True, config["plot_z_px_no_fp"], fp_ttl
    )

    # 7) RESOLVE OVERLAPPING CLUSTERS
    # Resolve overlapping water and non-water clusters.
    cluster_analysis.resolve_same_class_overlapping_clusters(shard_dict)
    cluster_analysis.resolve_diff_class_overlapping_clusters(shard_dict)
    plot_PCCs.plot_px_classification(shard_dict, config["plot_px_classification"])

    # 8) GENERATE REGRESSION MODEL
    # Generate a regression model for each telluric pixel.
    regression_model.find_regression_coeffs(shard_dict, calibrators, config)
    plot_regressions.plot_regressions(shard_dict, calibrators, config)

    # 9) WRITE MODEL TO DATABASE
    # Write out model to database.
    write_db.write_db(db_path, shard_dict, calibrators)

    # 10) ALERT THAT SPECTRUM IS COMPLETE
    # os.system('say "spectrum model complete"')


def create_tellurics(spectrum, model, pwv, order_wv_ranges, shards):
    # 2: Write telluric divided spectrum
    result = {}
    for order in range(len(spectrum)):
        if order in shards:
            # only one spectrum in shard
            spec = next(iter(shards[order].spectra.values()))

            # 2(a) Create tellurics array filled with nans
            tellurics = np.full(len(spectrum[order].wavelength), np.nan)

            # 2(b) Find the pixel at which the calculated telluric spectrum starts/
            # ends, and paste it into array
            l_cutoff_wv, r_cutoff_wv = order_wv_ranges[order]
            l_cutoff_px = np.argmax(spectrum[order].wavelength > l_cutoff_wv * u.AA)
            r_cutoff_px = np.argmax(spectrum[order].wavelength > r_cutoff_wv * u.AA)
            good_px = np.where(np.logical_not(np.isnan(spectrum[order].flux)))[0]
            l_cutoff_px = l_cutoff_px if l_cutoff_px > good_px[0] else good_px[0]
            r_cutoff_px = r_cutoff_px if r_cutoff_px < good_px[-1] else good_px[-1]
            tellurics[l_cutoff_px:r_cutoff_px] = spec.tel_lin_y

            # 2(c) Set telluric array in file
            result[order] = tellurics
        else:
            result[order] = np.ones(len(spectrum[order].wavelength))

    return result


def fit_selenite(science_spectrum, db_path, config_path):

    # 1) LOAD DATA
    # Load external data for calibration. External data is: i) configuration
    # file, ii) filename of spectrum to reduce, iii) content of calibration
    # spectra. Loads calibration spectra contents into a data container for
    # each order. These data containers are called shard_dict. Produces a
    # dictionary linking each processed order to its shard.
    config = yaml.safe_load(open(config_path, "r"))
    order = config["orders"]
    db, order_wv_ranges = read_db.read_db(db_path)

    shard_dict = {}
    spectrum = SpectrumList.read(science_spectrum)

    for orde in order:
        spectrum[orde] /= blaze[orde]
        spectrum[orde] /= np.nanpercentile(spectrum[orde].flux, 95)

    load_star(spectrum, shard_dict, None, order)

    plot_data.plot_data(shard_dict, "wv", "log", config["plot_raw_data"])

    # 2) CONSTRAIN SPECTRA WAVELENGTHS
    constrain_wvs.constrain_science_spectrum_wvs(shard_dict, order_wv_ranges)

    # 3) NORMALIZE DATA
    if config["normalize"]:
        normalize.continuum_normalize_all_orders(shard_dict)
    plot_data.plot_data(shard_dict, "wv", "lin", config["plot_norm_data"])

    # 4) FIND CALIBRATOR PX
    cal_pxs = telluric_id.find_calibrator_px(shard_dict, config)

    # 5) GENERATE TELLURIC SPECTRUM
    # i) Find mu by fitting the water calibrators' intensity to the sci. spectrum
    # ii) Retrieve z from science spectrum.
    # iii) Generate telluric spectrum for choice of mu and z
    # iv) Generate PWV metric
    mu = fit_model.get_mu(cal_pxs, shard_dict, db, config)
    z = fit_model.get_z(shard_dict)
    model = generate_model.generate_model(mu, z, db, config["saturation_threshold"])
    generate_model.generate_telluric_spectrum(shard_dict, model)
    pwv = generate_pwv_metric.generate_pwv_metric(shard_dict)
    plot_model.plot_model(
        shard_dict, model, pwv, "wavelength", show=config["plot_fit_telluric_spec"]
    )

    # 6) WRITE OUT MODEL
    tellurics = create_tellurics(spectrum, model, pwv, order_wv_ranges, shard_dict)

    for order, tell in tellurics.items():
        x = np.arange(len(tell))
        tellurics[order] = np.interp(x, x[tell > 0], tell[tell > 0])

    # 7) ALERT THAT SPECTRUM IS COMPLETE
    # os.system('say "spectrum reduction complete"')
    return tellurics


if __name__ == "__main__":
    load_store.load_star = load_star

    local = dirname(__file__)
    science_dir = join(local, "noise_1", "raw")
    medium_dir = join(local, "noise_1", "medium")
    science_file = join(local, "noise_1", "raw", "HD209458_b_32.fits")
    config_file = join(local, "noise_1", "calibrate_selenite_cfg.yml")
    fit_cfg_file = join(local, "noise_1", "fit_selenite_cfg.yml")

    database_file = join(local, "noise_1", "medium", "selenite_db.csv")

    detector = Crires("H/1/4", [1, 2, 3])
    blaze = detector.blaze
    wrange = detector.regions

    spectra = SpectrumArray.read(join(medium_dir, "spectra.npz"))
    star = spectra.meta["star"]
    observatory = detector.observatory
    times = spectra.datetime

    # normalize_bstars(config_file, spectra, join(medium_dir, "selenite"))
    calibrate_selenite(science_file, database_file, config_file)

    tell = fit_selenite(science_file, database_file, fit_cfg_file)

    spectrum = spectra[32][6]
    spectrum /= blaze[6]
    spectrum /= np.nanpercentile(spectrum.flux, 95)

    telluricmodel = SpectrumArray.read(join(medium_dir, "telluric.npz"))
    telluricmodel = telluricmodel[32][6]

    plt.plot(spectrum.wavelength, spectrum.flux, label="Spectrum")
    plt.plot(spectrum.wavelength, tell[6], label="Selenite")
    plt.plot(telluricmodel.wavelength, telluricmodel.flux, label="Input Tellurics")
    plt.legend()
    plt.show()

    pass
