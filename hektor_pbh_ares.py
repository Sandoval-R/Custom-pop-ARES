"""
Hektor's model for the accretion and energy injection
"""

import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

import ares
from ares.physics.Constants import erg_per_ev, k_B, m_p, c

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MSUN_G = 1.989e33 # Solar mass [g]
RY_EEV = 13.6 # Hydrogen ionization energy [eV]
MU_NEUTRAL = 1.22 # Mean molecular weight
GAMMA_AD = 5.0 / 3.0 # Adiabatic index
DEFAULT_MDOT_CRIT = 1.0e-2 # Critical value for dot{m}
L_EDD_PER_MSUN = 1.3e38 # Hektor's L_Edd / M [erg/s/Msun], below Eq. (2)

# ---------------------------------------------------------------------------
# Importing the SED
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SED_PATH = os.path.join(_HERE, 'hektor_low_mdot_sed.dat')
# Digitalized directly from Hektor's fig1,though it's a bit irrelevant in this process because it's normalized.'

# ---------------------------------------------------------------------------
# Cosmology, might substitute it for ARES later
# ---------------------------------------------------------------------------
DEFAULT_COSM_KWARGS = dict(
    cosmology_name='planck_TTTEEE_lowl_lowE',
    cosmology_id='best',
)
# Actually now it's just ARES, I need to clean a bit

# ---------------------------------------------------------------------------
# ADAF spectral template
# ---------------------------------------------------------------------------
class ADAFTemplate:
    """
    Loads a digitized, pre-normalized f(E) [1/eV] and interpolates in
    log-log space.
    """
    def __init__(self, path=DEFAULT_SED_PATH):
        data = np.loadtxt(path)
        E_eV, f_per_eV = data[:, 0], data[:, 1]

        # sanity check normalization (int f(E) dE = 1)
        norm = np.trapz(f_per_eV, E_eV)
        if not np.isclose(norm, 1.0, rtol=1e-3):
            raise ValueError(
                f"SED at {path} does not integrate to 1 (got {norm:.4f}); "
                "check the file or renormalize before using it."
            )

        self.E_min_eV = E_eV.min()
        self.E_max_eV = E_eV.max()
        self._log_interp = interp1d(
            np.log(E_eV), np.log(f_per_eV),
            kind='linear', bounds_error=False, fill_value=-np.inf,
        )

    def __call__(self, E_eV):
        E_eV = np.asarray(E_eV, dtype=float)
        out = np.exp(self._log_interp(np.log(np.clip(E_eV, 1e-30, None))))
        # outside the digitized range: zero
        out = np.where((E_eV < self.E_min_eV) | (E_eV > self.E_max_eV), 0.0, out)
        return out


class ConstantMass:
    """M_PBH(z) = M for all z [Msun]: Hektor's model, no cloud absorption."""
    def __init__(self, M_Msun):
        self.M_Msun = float(M_Msun)

    def __call__(self, z):
        return self.M_Msun


# ---------------------------------------------------------------------------
# PBH accretion model
# ---------------------------------------------------------------------------
class PBHAccretionModel:
    def __init__(
        self,
        M_PBH_Msun = 10.0, # Mass of PBH in solar masses
        f_PBH = 1.0, # Fraction of dark matter made of PBH.
        lam = 0.01, # Accretion parameter, value from Hektor
        beta = 0.2, # Exponent parameter in efficiency
        f_E = 0.15, # Fraction of emitted energy treated as absorbed/deposited. Hektor calculates .12 but uses .15
        mdot_crit = DEFAULT_MDOT_CRIT,
        cosm_kwargs = None,
        sed_path = DEFAULT_SED_PATH,
        mass_history = None, # M_PBH(z) [Msun]; constant M_PBH_Msun if None
    ):
        # Validation of parameters
        if M_PBH_Msun <= 0:
            raise ValueError("M_PBH_Msun must be > 0.")
        if not (0.0 <= f_PBH <= 1.0):
            raise ValueError("f_PBH must lie in [0, 1].")
        if lam < 0:
            raise ValueError("lam must be >= 0.")
        if not (0. <= beta <= 1.0):
            raise ValueError("Hektor et al. use beta in [0, 1].")
        if not (0.0 <= f_E <= 1.0):
            raise ValueError("f_E must lie in [0, 1].")
        if mdot_crit <= 0:
            raise ValueError("mdot_crit must be > 0.")

        # Storing parameters
        #self.M_PBH = float(M_PBH_Msun)
        #self.M_PBH_g = self.M_PBH * MSUN_G
        self.M_seed = float(M_PBH_Msun)
        self.mass_history = (ConstantMass(self.M_seed) if mass_history is None
                             else mass_history)
        self.f_PBH = float(f_PBH)
        self.lam = float(lam)
        self.beta = float(beta)
        self.f_E = float(f_E)
        self.mdot_crit = float(mdot_crit)

        kwargs = dict(DEFAULT_COSM_KWARGS) # Might be unnecessary, I'm almost sure that this is ARES default but it's a remnant from first idea of code.
        kwargs.update(cosm_kwargs or {})
        self.cosm = ares.physics.Cosmology(**kwargs)

        self.template = ADAFTemplate(sed_path)


        # Mass
    def M_PBH(self, z):
        """Instantaneous PBH mass [Msun]."""
        return float(self.mass_history(z))

    def L_Edd(self, z):
        """
        Eq. (2): L_Edd in erg/s, the value Hektor gives below Eq. (2),
        rather than the full 4 pi G M m_p c / sigma_T.
        """
        return L_EDD_PER_MSUN * self.M_PBH(z)


    # Densities
    def n_B(self, z):
        """Proper baryon number density [cm^-3], Eq. under Eq. (3) """
        rho_b = self.cosm.MeanBaryonDensity(z) # ARES function, gives rho_b (z)
        return rho_b / (MU_NEUTRAL * m_p)

    def n_H(self, z):
        """Proper hydrogen-nuclei density [cm^-3]."""
        return self.cosm.nH(z)

    def n_PBH(self, z):
        """Proper PBH number density [cm^-3], Eq. (4), from the seed mass.
        Still unsure about having it evolve with M(z)."""
        rho_cdm0 = self.cosm.omega_cdm_0 * self.cosm.CriticalDensity(0.0) # DM mass today
        rho_pbh = self.f_PBH * rho_cdm0 * (1.0 + z) ** 3
        return rho_pbh / (self.M_seed * MSUN_G)

    # -- Accretion physics ----------------------------------------------
    def sound_speed_km_s(self, Tk):
        Tk = float(Tk)
        if Tk <= 0.0:
            raise ValueError("Tk must be > 0 K.")
        cs_cgs = np.sqrt(GAMMA_AD * k_B * Tk / (MU_NEUTRAL * m_p))
        return cs_cgs / 1.0e5

    def sigma_vB_km_s(self, z, z_rec=1090.0):
        return (30.0 / np.sqrt(3.0)) * (1.0 + z) / (1.0 + z_rec)

    def v_eff_km_s(self, z, Tk):
        sigma = float(self.sigma_vB_km_s(z))
        cs = float(self.sound_speed_km_s(Tk))
        exponent = -3.0 * (1.0 + self.beta) / 2.0

        def integrand(v):
            fv = (
                np.sqrt(2.0 / np.pi) * v**2
                * np.exp(-v**2 / (2.0 * sigma**2)) / sigma**3
            )
            return (v**2 + cs**2)**exponent * fv

        average, _ = quad(integrand, 0.0, np.inf, epsabs=0.0,
                           epsrel=1e-9, limit=200)
        return average ** (-1.0 / (3.0 * (1.0 + self.beta)))

    def mdot(self, z, Tk):
        """
        Aproximation for dot{m}, eq. 3
        """
        nB = self.n_B(z)
        veff = self.v_eff_km_s(z, Tk)
        return (
            8.0e-7 * self.lam * (self.M_PBH(z) / 10.0)
            * (nB / 1.0) * (veff / 10.0) ** -3.0
        )

    def efficiency(self, z, Tk):
        """
        Defining eta, eq. 1
        """
        m = self.mdot(z, Tk)
        ratio = m / self.mdot_crit
        return 0.1 * (ratio ** self.beta if ratio <= 1.0 else 1.0)

    def L_bol(self, z, Tk):
        """erg/s. For mdot <= mdot_crit (always true per the paper),
        L = L_Edd * mdot * (mdot/mdot_crit)^beta -- Eq. (2) integrated."""
        m = self.mdot(z, Tk)
        eta = self.efficiency(z, Tk)
        return self.L_Edd(z) * m * (eta / 0.1)

    def L_E(self, E_eV, z, Tk):
        """
        Specific luminosity, eq.2
        erg/s/eV.
        """
        return self.L_bol(z, Tk) * self.template(E_eV)

    def j_E(self, E_eV, z, Tk):
        """
        Spectral emissivity, Eq. (4).
        erg/s/cm^3/eV
        """
        return self.n_PBH(z) * self.L_E(E_eV, z, Tk)

    def epsilon(self, z, Tk):
        """
        Absorbed energy rate density [erg/s/cm^3], Eq. (6).
        Since the template is normalized (int f dE = 1), this reduces analytically
        to f_E * n_PBH * L_bol -- use epsilon_integrated() as an
        independent check if you want to verify that explicitly.
        """
        return self.f_E * self.n_PBH(z) * self.L_bol(z, Tk)

    def epsilon_integrated(self, z, Tk):
        """Diagnostic: same as epsilon(), by explicit SED integration."""
        val, _ = quad(
            lambda E: float(self.j_E(E, z, Tk)),
            self.template.E_min_eV, self.template.E_max_eV,
            epsabs=0.0, epsrel=1e-6, limit=400,
        )
        return self.f_E * val

    # -- ARES-facing rate coefficients (per absorbing atom) --

    """
        This section relies on another paper https://doi.org/10.1016/j.physletb.2018.09.009.
        It states that
        f_heat = (1+2x_e)/3
        f_ion = (1-x_e)/3
    """

    def n_HI(self, z, xe):
        return self.n_H(z) * max(1.0 - float(xe), 1e-30)

    def heat_rate_coefficient(self, z, Tk, xe):
        """erg/s per neutral H atom."""
        f_heat = (1.0 + 2.0 * float(xe)) / 3.0
        return f_heat * self.epsilon(z, Tk) / self.n_HI(z, xe)

    def ion_rate_coefficient(self, z, Tk, xe):
        """s^-1 per neutral H atom."""
        f_ion = (1.0 - float(xe)) / 3.0
        ion_rate_density = f_ion * self.epsilon(z, Tk) / (RY_EEV * erg_per_ev)
        return ion_rate_density / self.n_HI(z, xe)

#---
def make_stateful_callbacks(model, Tk_of_z=None, xe_of_z=None):
    """
    ARES heating and ionization callbacks reading the live IGM state, which
    hektor_ares_patch forwards to callbacks marked _ares_stateful.

    ARES passes igm_Tk and igm_h_1 but not igm_e, so x_e is the hydrogen
    ionization state 1 - igm_h_1.  Tk_of_z and xe_of_z serve direct calls
    outside ARES; without them a call lacking the live state is an error
    rather than a silent fallback.
    """
    def live_state(z, kwargs):
        Tk = kwargs.get('igm_Tk')
        if Tk is None:
            if Tk_of_z is None:
                raise RuntimeError('PBH callback called without igm_Tk at '
                    'z = {:g}.'.format(z))
            Tk = Tk_of_z(z)

        h1 = kwargs.get('igm_h_1')
        if h1 is not None:
            xe = 1.0 - h1
        elif xe_of_z is not None:
            xe = xe_of_z(z)
        else:
            raise RuntimeError('PBH callback called without igm_h_1 at '
                'z = {:g}.'.format(z))
        return float(Tk), float(xe)

    # ARES asks for a coefficient per absorbing species (0 = HI, 1 = HeI,
    # 2 = HeII) and multiplies each by that species' density.  Hektor's
    # deposition is all charged to HI: heat (1+2x_e)/3 warms all the gas
    # through ARES's heat capacity, ionization (1-x_e)/3 goes to hydrogen.
    # The helium species get nothing, or the energy would count twice.
    def heat_rate(z, species=0, **kwargs):
        """ARES heat-rate callback [erg/s per neutral H atom]."""
        if species != 0:
            return 0.0
        return model.heat_rate_coefficient(z, *live_state(z, kwargs))

    def ion_rate_igm(z, species=0, **kwargs):
        """ARES ionization callback [1/s per neutral H atom]."""
        if species != 0:
            return 0.0
        return model.ion_rate_coefficient(z, *live_state(z, kwargs))

    # The runtime ARES adapter checks this marker before forwarding the live
    # IGM state into the callback.
    heat_rate._ares_stateful = True
    ion_rate_igm._ares_stateful = True
    return heat_rate, ion_rate_igm


# ---------------------------------------------------------------------------
# ARES ParametricPopulation adapter
# ---------------------------------------------------------------------------
def make_pbh_population(
    M_PBH_Msun=10.0, f_PBH=1.0, lam=0.01, beta=0.2, f_E=0.15,
    mdot_crit=DEFAULT_MDOT_CRIT, Tk_of_z=None, xe_of_z=None,
    cosm_kwargs=None, sed_path=DEFAULT_SED_PATH, pop_extra=None,
    mass_history=None,
):
    model = PBHAccretionModel(
        M_PBH_Msun=M_PBH_Msun, f_PBH=f_PBH, lam=lam, beta=beta, f_E=f_E,
        mdot_crit=mdot_crit, cosm_kwargs=cosm_kwargs, sed_path=sed_path,
        mass_history=mass_history,
    )

    heat_rate, ion_rate_igm = make_stateful_callbacks(model, Tk_of_z, xe_of_z)

    pars = dict(
        pop_sfr_model='hektor2018',
        pop_heat_rate=heat_rate,
        pop_ion_rate_igm=ion_rate_igm,
        pop_Ja=None, # Will be interesting to turn this on, but needs tinkering with several things.
        pop_zform=100.0, # Not when it's turn on
        pop_zdead=6.0,
    )
    if pop_extra:
        pars.update(pop_extra)

    return pars, model

# -- Small test--
if __name__ == '__main__':
    model = PBHAccretionModel(M_PBH_Msun=10., f_PBH=1., lam=0.01, beta=0.2, f_E=0.15)
    for z in [30., 20., 17., 10.]:
        Tk = 2.725 * (1. + z)  # crude smoke-test Tk only
        print(f"z={z:5.1f}  mdot={model.mdot(z, Tk):.3e}  "
              f"Lbol={model.L_bol(z, Tk):.3e} erg/s  "
              f"epsilon={model.epsilon(z, Tk):.3e} erg/s/cm^3")
