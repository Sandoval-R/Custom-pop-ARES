"""
Adapter for the combined model.

It takes everything from hektor_pbh_ares and just replaces where the M_PBH varying mass affects.
"""
import numpy as np

from hektor_pbh_ares import (
    PBHAccretionModel as HektorPBHAccretionModel,
    DEFAULT_MDOT_CRIT,
    DEFAULT_SED_PATH,
)
from d2b_model import DarkToBlack


SEC_PER_YR = 365.25 * 24.0 * 3600.0

# PBHAccretionModel inherits from HektorPBHAccretionModel
# Thas is, I will access Hektor's parameters with super()
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
        d2b_z_end=5.0,
        d2b_enabled=True, # For test verifying if the new accretion gives relevant results.
        d2b_cloud_floor_Msun=1.0e-15,
        d2b_rtol=1.0e-9,
        d2b_atol=1.0e-12,
        d2b_method="RK45",
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

        # The Hektor base class stores the seed mass in self.M_PBH.  Keep that seed explicitly because n_PBH might nedd to continue to use the initial population mass, while M_PBH(z) below becomes the instantaneous D2B mass.
        self.M_PBH_seed = float(self.M_PBH)
        del self.M_PBH

        self.d2b_z_start = float(d2b_z_start)
        self.d2b_z_end = float(d2b_z_end)

        # Condition for activating D2B accretion
        self.d2b_enabled = bool(d2b_enabled) and d2b_Mcloud_Msun > 0.0

        # Initialize attributes that are relevant for D2B. If it is off, the code can still read the None.
        self.d2b = None
        self.d2b_t0_yr = None
        self.d2b_tend_yr = None
        self.t_depletion_yr = None
        self.z_depletion = None

        if not self.d2b_enabled:
            return

        if self.d2b_z_start <= self.d2b_z_end:
            raise ValueError("d2b_z_start must be greater than d2b_z_end.")

        # ARES provides the cosmological time-redshift relation.  D2B itself remains entirely in the tested time-domain implementation.
        self.d2b_t0_yr = float(
            self.cosm.t_of_z(self.d2b_z_start) / SEC_PER_YR
        )
        self.d2b_tend_yr = float(
            self.cosm.t_of_z(self.d2b_z_end) / SEC_PER_YR
        )

        self.d2b = DarkToBlack(
            mu_eV=d2b_mu_eV,
            Mbh0=self.M_PBH_seed,
            Mc0=d2b_Mcloud_Msun,
            t0_yr=self.d2b_t0_yr, # z_initial in years
            t_end_yr=self.d2b_tend_yr,
            cloud_floor=d2b_cloud_floor_Msun,
            rtol=d2b_rtol,
            atol=d2b_atol,
            method=d2b_method,
        )

        # solve() builds the adaptive dense solution used by state_at().
        history = self.d2b.solve(n_output=2)
        self.t_depletion_yr = history["t_depletion_yr"]

        if self.t_depletion_yr is not None:
            self.z_depletion = float(
                self.cosm.z_of_t(self.t_depletion_yr * SEC_PER_YR)
            )

    # ------------------------------------------------------------------
    # D2B -> Hektor integration
    # ------------------------------------------------------------------
    # Create M_PBH as a variable
    def M_PBH(self, z):
        """Instantaneous PBH mass entering the Hektor equations [Msun]."""
        z = np.asarray(z, dtype=float)

        # Fallback if D2B is deactivated, basically it fills the array with a constant mass.
        if self.d2b is None:
            return np.full_like(z, self.M_PBH_seed, dtype=float)

        t_yr = np.asarray(self.cosm.t_of_z(z), dtype=float) / SEC_PER_YR
        return self.d2b.state_at(t_yr)["Mbh"]

    # Only the two Hektor equations that explicitly contain M_PBH need a correction. Everything comes directly from hektor_pbh_ares.py.

    def mdot(self, z, Tk):
        """Hektor Eq. (3), evaluated with the instantaneous D2B mass."""
        nB = super().n_B(z) # Comes from Hektor
        veff = super().v_eff_km_s(z, Tk) # Comes from Hektor
        M = float(self.M_PBH(z))

        return (
            8.0e-7 * self.lam * (M / 10.0)
            * (nB / 1.0) * (veff / 10.0) ** -3.0
        )

    def L_bol(self, z, Tk):
        """Hektor Eq. (2), evaluated with the instantaneous D2B mass."""
        M = float(self.M_PBH(z))
        L_Edd = 1.3e38 * M
        m = self.mdot(z, Tk)
        eta = self.efficiency(z, Tk)
        return L_Edd * m * (eta / 0.1)


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
    d2b_z_end=6.0,
    d2b_enabled=True,
    d2b_cloud_floor_Msun=1.0e-15,
    d2b_rtol=1.0e-9,
    d2b_atol=1.0e-12,
    d2b_method="RK45",
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
        d2b_z_end=d2b_z_end,
        d2b_enabled=d2b_enabled,
        d2b_cloud_floor_Msun=d2b_cloud_floor_Msun,
        d2b_rtol=d2b_rtol,
        d2b_atol=d2b_atol,
        d2b_method=d2b_method,
    )

    if Tk_of_z is None:
        Tk_of_z = model.cosm.Tgas

    if xe_of_z is None:
        def xe_of_z(z, **kw):
            h1 = kw.get("igm_h_1")
            if h1 is not None:
                return 1.0 - h1
            return 2.0e-4

    def heat_rate(z, **kwargs):
        Tk = kwargs.get("igm_Tk")
        if Tk is None:
            Tk = Tk_of_z(z)

        xe = kwargs.get("igm_e")
        if xe is None:
            xe = xe_of_z(z)

        return model.heat_rate_coefficient(z, float(Tk), float(xe))

    def ion_rate_igm(z, **kwargs):
        Tk = kwargs.get("igm_Tk")
        if Tk is None:
            Tk = Tk_of_z(z)

        xe = kwargs.get("igm_e")
        if xe is None:
            xe = xe_of_z(z)

        return model.ion_rate_coefficient(z, float(Tk), float(xe))

    heat_rate._ares_stateful = True
    ion_rate_igm._ares_stateful = True

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
