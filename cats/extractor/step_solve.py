from os.path import join

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from astropy import units as u

from ..solver.bayes import BayesSolver
from ..solver.linear import LinearSolver
from ..solver.spline import SplineSolver
from .steps import Step, StepIO


class PlanetAbsorptionAreaStep(Step, StepIO):
    filename = "planet_area.npy"

    def run(self, spectra):
        limits = 15, 85
        f = spectra.flux.to_value(1)
        y = np.nanmean(f, axis=1)
        x = np.arange(len(y))

        x2 = np.concatenate([x[: limits[0]], x[limits[1] :]])
        y2 = np.concatenate([y[: limits[0]], y[limits[1] :]])
        yf = np.polyval(np.polyfit(x2, y2, 3), x)

        area = 1 - y / yf
        area[: limits[0]] = area[limits[1] :] = 0
        area = gaussian_filter1d(area, 1)
        self.save(area)
        return area

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        np.save(filename, data)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = np.load(filename)
        return data


class SolveProblemStep(Step):
    def solve_prepared(
        self,
        spectra,
        telluric,
        stellar,
        intensities,
        detector,
        star,
        planet,
        seg=5,
        solver="linear",
        rv=None,
        regularization_ratio=1,
        regularization_weight=None,
        n_sysrem=None,
        area=None,
    ):
        # regweight:
        # for noise 0:  1
        # for noise 1%: 23
        # print("Solving the problem...")
        spectra = spectra.get_segment(seg)
        telluric = telluric.get_segment(seg)
        stellar = stellar.get_segment(seg)
        intensities = intensities.get_segment(seg)

        times = spectra.datetime
        wavelength = spectra.wavelength.to_value(u.AA)

        spectra = spectra.flux.to_value(1)
        telluric = telluric.flux.to_value(1)
        stellar = stellar.flux.to_value(1)
        intensities = intensities.flux.to_value(1)

        if solver == "linear":
            solver = LinearSolver(
                detector,
                star,
                planet,
                regularization_ratio=regularization_ratio,
                plot=False,
                regularization_weight=regularization_weight,  # 0.01,
                method="Tikhonov",
                n_sysrem=n_sysrem,
            )
        elif solver == "spline":
            solver = SplineSolver(detector, star, planet)
        elif solver == "bayesian":
            solver = BayesSolver(detector, star, planet)
        else:
            raise ValueError(
                "Unrecognized solver option {solver} expected one of ['linear', 'spline', 'bayesian']"
            )

        spec = solver.solve(
            times, wavelength, spectra, stellar, intensities, telluric, rv=rv, area=area
        )
        solver.regularization_weight = spec.meta["regularization_weight"]

        return spec

    def run(
        self,
        normalized_observation,
        planet_reference_spectrum,
        spectra,
        detector,
        star,
        planet,
        planet_radial_velocity,
        planet_area,
    ):
        normalized, broadening = normalized_observation
        v_planet = planet_radial_velocity["rv_planet"]
        wavelength = normalized[51].wavelength
        flux = normalized[51].flux

        planet_reference_spectrum.flux[:] = gaussian_filter1d(
            planet_reference_spectrum.flux, 30
        )

        return_data = {}

        segments = self.segments
        if segments == "all":
            nseg = spectra.nseg
            segments = range(nseg)

        for seg in tqdm(segments):
            hspec = planet_reference_spectrum.resample(wavelength[seg], inplace=False)
            hspec.flux[:] -= np.nanmin(hspec.flux)
            hspec.flux[:] /= np.nanmax(hspec.flux)

            data = [
                {
                    "x": wavelength[seg].to_value("AA"),
                    "y": flux[seg].to_value(1),
                    "name": "normalized observation",
                },
                {
                    "x": wavelength[seg].to_value("AA"),
                    "y": hspec.flux.to_value(1),
                    "name": "planet model",
                },
            ]

            solver = self.solver
            regularization_weight = self.regularization_weight
            sysrem_iterations = self.sysrem_iterations

            spec = self.solve_prepared(
                spectra,  # (normalized) observation
                spectra,  # Stellar spectrum
                spectra,  # Tellurics
                spectra,  # specific intensities
                detector,
                star,
                planet,
                solver=solver,
                seg=seg,
                rv=v_planet,
                n_sysrem=sysrem_iterations,
                regularization_weight=regularization_weight,
                regularization_ratio=10,
                area=planet_area,
            )

            return_data[seg] = spec

            swave = spec.wavelength.to_value("AA")
            sflux = spec.flux.to_value(1)
            sflux = sflux - np.nanpercentile(sflux, 5)
            sflux /= np.nanpercentile(sflux, 95)

            data += [
                {
                    "x": swave,
                    "y": sflux,
                    "name": f"extracted, RegWeight: {regularization_weight}, nSysrem: {sysrem_iterations}",
                },
            ]

            wran = [
                wavelength[seg][0].to_value("AA"),
                wavelength[seg][-1].to_value("AA"),
            ]
            layout = {
                "title": f"Segment: {seg}",
                "xaxis": {"title": "Wavelength [Å]", "range": wran},
                "yaxis": {"title": "Flux, normalised"},
            }
            fname = join(self.done_dir, f"planet_spectrum_{seg}.html")
            fig = go.Figure(data, layout)
            py.plot(fig, filename=fname, auto_open=False)

        return return_data
