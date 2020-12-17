from ..data_modules.telluric_model import TelluricModel
from ..data_modules.telluric_tapas import TapasTellurics
from ..data_modules.space import Space

from .steps import Step
from ..spectrum import SpectrumArrayIO


class TelluricStep(Step, SpectrumArrayIO):
    filename = "telluric.npz"
    source = None

    def run(self, spectra, star, observatory, detector):
        times = spectra.datetime
        wrange = detector.regions
        reference_frame = spectra.reference_frame

        telluric = self.get_datasource(source, star, observatory)
        spec = telluric.get(wrange, times)
        if spec.reference_frame != reference_frame:
            spec = spec.shift(reference_frame, inplace=True)
        spec = spec.resample(spectra.wavelength, inplace=False, method="linear")
        spec.segments = np.copy(spectra.segments)

        self.save(data, self.savefilename)
        return data

    def get_datasource(self, source, star, observatory):
        if source == "model":
            telluric = TelluricModel(star, observatory)
        elif source == "space":
            telluric = Space()
        elif source == "tapas":
            telluric = TapasTellurics(star, observatory)
        else:
            raise ValueError(
                f"Expected one of ['model', 'space', 'tapas'] but got {source} instead"
            )
        return telluric


class TelluricAirmassStep(TelluricStep):
    filename = "telluric_airmass.npz"
    source = "model"


class TelluricTapasStep(TelluricStep):
    filename = "telluric_tapas.npz"
    source = "tapas"


class TelluricSpaceStep(TelluricStep):
    filename = "telluric_space.npz"
    source = "space"

