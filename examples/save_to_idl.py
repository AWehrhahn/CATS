from pysme.sme import SME_Structure
from pysme.persistence import save_as_idl

from os.path import dirname, join

sme = SME_Structure.load(
    join(dirname(__file__), "noise_5", "medium", "spectrum_with_mask.sme")
)
sme.atmo.geom = "SPH"
sme.atmo.source = "marcs2012s_t5.0.sav"
save_as_idl(sme, join(dirname(__file__), "noise_5", "medium", "spec.inp"))
