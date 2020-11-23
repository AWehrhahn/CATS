import contextlib
import sys

import numpy as np
from tqdm import tqdm
from astropy import units


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


@units.quantity_input(wl_air=units.AA)
def air2vac(wl_air, copy=True):
    """
    Convert wavelengths in air to vacuum wavelength
    Author: Nikolai Piskunov

    Note: works only for wavelengths above 2000 Angstrom
    """
    if copy:
        wl_vac = np.copy(wl_air)
    else:
        wl_vac = wl_air

    ii = wl_air > 1999.352 * units.AA

    # Compute wavenumbers squared
    sigma2 = (1e4 / wl_vac[ii].to_value(units.AA)) ** 2
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    # Convert to vacuum wavelength
    wl_vac[ii] *= fact

    return wl_vac


@units.quantity_input(wl_vac=units.AA)
def vac2air(wl_vac, copy=True):
    """
    Convert vacuum wavelengths to wavelengths in air
    Author: Nikolai Piskunov

    Note: works only for wavelengths below 20000 Ansgtrom
    """
    if copy:
        wl_air = np.copy(wl_vac)
    else:
        wl_air = wl_vac

    ii = wl_vac > 20_000 * units.AA

    # Compute wavenumbers squared
    sigma2 = (1e4 / wl_air[ii].to_value(units.AA)) ** 2
    fact = (
        1e0
        + 8.34254e-5
        + 2.406147e-2 / (130e0 - sigma2)
        + 1.5998e-4 / (38.9e0 - sigma2)
    )
    # Convert to air wavelength
    wl_air[ii] /= fact
    return wl_air
