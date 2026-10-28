"""
Adapter for the combined model.

It takes everything from hektor_pbh_ares and just replaces where the M_PBH varying mass affects.
"""
import numpy as np

from hektor_pbh_ares import (
    PBHAccretionModel as HektorPBHAccretionModel,
    DEFAULT_MDOT_CRIT,
    DEFAULT_SED_PATH,
    make_stateful_callbacks,
)
from d2b_model import Mass_analytic


SEC_PER_YR = 365.25 * 24.0 * 3600.0

# PBHAccretionModel inherits from HektorPBHAccretionModel
# That is, I will access Hektor's parameters with super()
class PBHAccretionModel(HektorPBHAccretionModel):
    """Hektor PBH model with an optional Dark-to-Black mass history."""

    def __init__(
        self,
        M_PBH_Msun=10.0,
        f_PBH=1.0,
        lam=0.01,
        beta=0.2,
        f_E=0.15,
        mdot_crit=DEFAULT_MDOT_CRIT,
        cosm_kwargs=None,
        sed_path=DEFAULT_SED_PATH,
        d2b_mu_eV=1.73e-17,
        d2b_Mcloud_Msun=0.0,
        d2b_z_start=60.0,
        d2b_enabled=True, # For test verifying if the new accretion gives relevant results.
    ):
        # Let the tested Hektor implementation initialize all of its accretion, SED, density, and ARES-related quantities unchanged.
        super().__init__(
            M_PBH_Msun=M_PBH_Msun,
            f_PBH=f_PBH,
            lam=lam,
            beta=beta,
            f_E=f_E,
            mdot_crit=mdot_crit,
            cosm_kwargs=cosm_kwargs,
            sed_path=sed_path,
        )

        self.d2b_z_start = float(d2b_z_start)

        # Condition for activating D2B accretion
        self.d2b_enabled = bool(d2b_enabled) and d2b_Mcloud_Msun > 0.0

        # Initialize attributes that are relevant for D2B. If it is off, the code can still read the None.
        self.d2b = None
        self.z_depletion = None

        if not self.d2b_enabled:
            return

        # The draft's closed-form M_PBH(z), starting at d2b_z_start.
        self.d2b = Mass_analytic(
            self.M_seed, d2b_Mcloud_Msun, d2b_mu_eV, self.d2b_z_start,
            self.cosm,
        )
        self.z_depletion = self.d2b.z_depletion

        # n_PBH keeps using the seed mass; the D2B mass drives Bondi accretion and L_Edd.
        self.mass_history = self.d2b

# ---------------------------------------------------------------------------
# ARES population factory
# ---------------------------------------------------------------------------

def make_pbh_population(
    M_PBH_Msun=10.0,
    f_PBH=1.0,
    lam=0.01,
    beta=0.2,
    f_E=0.15,
    mdot_crit=DEFAULT_MDOT_CRIT,
    Tk_of_z=None,
    xe_of_z=None,
    cosm_kwargs=None,
    sed_path=DEFAULT_SED_PATH,
    d2b_mu_eV=1.73e-17,
    d2b_Mcloud_Msun=0.0,
    d2b_z_start=60.0,
    d2b_enabled=True,
    pop_extra=None,
):
    """Construct the Hektor+D2B source and its ARES callbacks."""

    model = PBHAccretionModel(
        M_PBH_Msun=M_PBH_Msun,
        f_PBH=f_PBH,
        lam=lam,
        beta=beta,
        f_E=f_E,
        mdot_crit=mdot_crit,
        cosm_kwargs=cosm_kwargs,
        sed_path=sed_path,
        d2b_mu_eV=d2b_mu_eV,
        d2b_Mcloud_Msun=d2b_Mcloud_Msun,
        d2b_z_start=d2b_z_start,
        d2b_enabled=d2b_enabled,
    )

    heat_rate, ion_rate_igm = make_stateful_callbacks(model, Tk_of_z, xe_of_z)

    pars = dict(
        pop_sfr_model="hektor2018",
        pop_heat_rate=heat_rate,
        pop_ion_rate_igm=ion_rate_igm,
        pop_Ja=None,
        pop_zform=100.0,
        pop_zdead=6.0,
    )

    if pop_extra:
        pars.update(pop_extra)

    return pars, model
