"""
Dark-to-Black (D2B) mass history of one PBH seed absorbing its scalar cloud.

    DarkToBlack           the history in cosmic time [yr], by ODE
    Mass_analytic         M_PBH(z) for ARES from the twentyoneSMBH draft
                          (hektor_pbh_ares.make_pbh_population(mass_history=...))

Model
-----
In the small-alpha limit the black hole absorbs the cloud at

    dM_BH/dt = |Gamma| M_c,   dM_c/dt = -|Gamma| M_c,
    |Gamma| = 16 alpha^6 c^3 / (G M_BH),   alpha = mu G M_BH / (hbar c^3),

so |Gamma| scales as M_BH^5.  M_BH + M_c is conserved, and with
u = ln(M_c0 / M_c) the system reduces exactly to one equation:

    M_c = M_c0 e^-u,   M_BH = M_BH,0 + M_c0 - M_c,   du/dt = |Gamma(M_BH)|.

Masses are in Msun, DarkToBlack's times in years.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================================
# Physical constants
# ============================================================================

G = 6.67430e-11                 # m^3 kg^-1 s^-2
c = 299792458.0                 # m s^-1
HBAR = 1.054571817e-34          # J s
EV_J = 1.602176634e-19          # J
M_SUN = 1.98847e30              # kg
YEAR = 365.25 * 24.0 * 3600.0   # s

ODE_METHODS = ("Radau", "BDF", "LSODA", "RK45", "DOP853")


# ============================================================================
# D2B rate
# ============================================================================

def alpha(Mbh, mu_eV):
    """
    Dimensionless gravitational coupling alpha = mu M_BH (G = c = hbar = 1).

    Restoring units, mu is supplied as an energy and converted to inverse
    metres, mu [1/m] = mu[eV] eV[J] / (hbar c), and the BH mass to its
    gravitational length, G M_BH / c^2 [m].
    """
    mu_inv_m = mu_eV * EV_J / (HBAR * c)
    M_geom_m = G * np.asarray(Mbh) * M_SUN / c**2 # To allow multiple masses
    return mu_inv_m * M_geom_m


def gamma_abs(Mbh, mu_eV):
    """
    Absorption rate |Gamma| [1/s].  In natural units |Gamma| = 16 alpha^6 / M;
    restoring units, |Gamma| = 16 alpha^6 c^3 / (G M_BH).  The cloud's
    e-folding time is 1 / |Gamma|.
    """
    M_kg = np.asarray(Mbh) * M_SUN
    return 16.0 * alpha(Mbh, mu_eV)**6 * c**3 / (G * M_kg)


# ============================================================================
# One black hole in cosmic time
# ============================================================================

class DarkToBlack:
    """
    Evolve one black hole embedded in a finite scalar cloud, from t0_yr on.

    Call solve() first.  The cloud counts as used up once M_c < cloud_floor;
    from then on the state is held there, and the cloud and accretion rate
    are reported as 0.
    """

    def __init__(
        self,
        mu_eV=1.73e-17, # mass of ultralight boson field associated particle
        Mbh0=1.0e3, # BH seed
        Mc0=1.0e6, # Initial cloud mass
        t0_yr=0.0, # Initial time of the simulation... might need to fix it to avoid confusing with ARES
        t_end_yr=1.5e9, # Basically a stopper... might change it later but it gives a safe stop right now
        cloud_floor=1.0e-15, # Acceptable zero mass for cloud
        rtol=1e-9, # Tolerance for the ODE solver / Might ask Dario's team'
        atol=1e-12, # Absolute tolerance / Might ask Dario's team'
        method="RK45", # One of ODE_METHODS (RK45 is Runge-Kutta)
    ):
        self.mu_eV = float(mu_eV)
        self.Mbh0 = float(Mbh0)
        self.Mc0 = float(Mc0)
        self.t0_yr = float(t0_yr)
        self.t_end_yr = float(t_end_yr)
        self.cloud_floor = float(cloud_floor)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.method = str(method)

        if self.mu_eV <= 0:
            raise ValueError("mu_eV must be positive.")
        if self.Mbh0 <= 0:
            raise ValueError("Mbh0 must be positive.")
        if self.Mc0 <= 0:
            raise ValueError("Mc0 must be positive.")
        if self.cloud_floor <= 0:
            raise ValueError("cloud_floor must be positive.")
        if self.cloud_floor >= self.Mc0:
            raise ValueError("cloud_floor must be smaller than Mc0.")
        if self.t_end_yr <= self.t0_yr:
            raise ValueError("t_end_yr must exceed t0_yr.")
        if self.method not in ODE_METHODS:
            raise ValueError("Unsupported method. Use one of "
                + ", ".join(ODE_METHODS) + ".")

        self.M_total = self.Mbh0 + self.Mc0 # Total mass of the system
        self.q = self.Mc0 / self.Mbh0 # Cloud ratio
        self.gamma0 = float(self.gamma_abs(self.Mbh0)) # Seed rate [1/s]

        # u at the numerical cloud-depletion threshold.
        self.u_final = np.log(self.Mc0 / self.cloud_floor)

        self._solution = None
        self.t_depletion_yr = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def solve(self, n_output=10000):
        """
        The state on n_output times evenly spaced from t0_yr to t_end_yr.
        It integrates first, up to depletion, and sets t_depletion_yr.
        """
        self._integrate()
        t_eval = np.linspace(self.t0_yr, self.t_end_yr, int(n_output))
        return self.state_at(t_eval)

    def state_at(self, t_yr):
        """
        The state at cosmic time(s) t_yr: a dict with t_yr, Mbh, Mc, mdot,
        alpha, t_efold_yr, u and t_depletion_yr.  Useful for the ARES
        integration.
        """
        t = np.asarray(t_yr, dtype=float)
        if self._solution is None:
            raise RuntimeError("Call solve() before state_at().")
        u = np.asarray(self._solution.sol(t)[0], dtype=float)

        # At and after depletion, hold the final state fixed.
        if self.t_depletion_yr is not None:
            depleted = t >= self.t_depletion_yr
            u = np.where(depleted, self.u_final, u)
        else:
            depleted = np.zeros_like(t, dtype=bool)

        # Guard against tiny numerical excursions beyond the event.
        u = np.clip(u, 0.0, self.u_final)

        # Reconstruct both masses exactly from the conserved total mass.
        Mc = self.Mc0 * np.exp(-u)
        Mbh = self.M_total - Mc

        # The floor counts as complete depletion for the accretion source
        # term, and the cloud is reported as zero.
        return {
            "t_yr": t,
            "Mbh": Mbh,
            "Mc": np.where(depleted, 0.0, Mc),
            "mdot": np.where(depleted, 0.0, self.mdot(Mbh, Mc)),
            "alpha": self.alpha(Mbh),
            "t_efold_yr": self.t_efold_yr(Mbh),
            "u": u,
            "t_depletion_yr": self.t_depletion_yr,
        }

    def Mbh_at(self, t_yr):
        """PBH mass [Msun] at cosmic time(s) t_yr."""
        return self.state_at(t_yr)["Mbh"]

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------

    def alpha(self, Mbh):
        """alpha = mu M_BH (see the module function)."""
        return alpha(Mbh, self.mu_eV)

    def gamma_abs(self, Mbh):
        """|Gamma| [1/s] (see the module function)."""
        return gamma_abs(Mbh, self.mu_eV)

    def mdot(self, Mbh, Mc):
        """Accretion rate dM_BH/dt = |Gamma| M_c [Msun/yr]."""
        return self.gamma_abs(Mbh) * np.asarray(Mc) * YEAR

    def t_efold_yr(self, Mbh):
        """Cloud e-folding time 1 / |Gamma| [yr]."""
        return 1.0 / self.gamma_abs(Mbh) / YEAR

    # ------------------------------------------------------------------
    # u(t), by ODE
    # ------------------------------------------------------------------

    def _integrate(self):
        """Integrate du/dt = |Gamma| up to depletion or t_end_yr."""
        sol = solve_ivp(
            self._rhs_u,
            (self.t0_yr, self.t_end_yr),
            np.array([0.0]),
            events = self._depletion_event_u,
            rtol = self.rtol,
            atol = self.atol,
            method = self.method,
            dense_output = True,
            max_step = 1.0e6,
        )
        if not sol.success:
            raise RuntimeError(
                "Dark-to-black integration failed: " + sol.message
            )
        self._solution = sol
        self.t_depletion_yr = (float(sol.t_events[0][0])
            if len(sol.t_events[0]) else None)

    def _rhs_u(self, t_yr, y):
        """du/dt = |Gamma(M_BH)| [1/yr], with M_BH = M_total - Mc0 e^-u."""
        u = max(float(y[0]), 0.0) # u must be positive for depleting cloud
        Mbh = self.M_total - self.Mc0 * np.exp(-u)
        return np.array([float(self.gamma_abs(Mbh)) * YEAR])

    def _depletion_event_u(self, t_yr, y):
        """Terminate at the numerical cloud-depletion threshold."""
        return float(y[0]) - self.u_final

    _depletion_event_u.terminal = True
    _depletion_event_u.direction = 1.0


# ============================================================================
# M_PBH(z) for ARES
# ============================================================================

class Mass_analytic:
    """
    M_PBH(z) [Msun] from the twentyoneSMBH draft (v5), which integrates
    dM_BH/dt = |Gamma| M_c with M_c held at its initial value M_c0:

        M(z) = M_0 [1 + K ln((1 + w_0)(1 - w) / ((1 - w_0)(1 + w)))]^(-1/4),
        w = sqrt(1 + a_1 (1 + z)^3),   w_0 = w(z_seed),
        K = 64 mu^6 M_c0 M_0^4 / (3 H_0 sqrt(Omega_L)),   a_1 = Omega_m / Omega_L.

    The log equals -2 (F(z) - F(z_seed)) with
    F(z) = arccoth sqrt(1 + a_1 (1 + z)^3), which is what the code uses.
    Three choices differ from the draft's text: z_0 = z_seed is where the
    absorption starts (M(z_seed) = M_0), a_1 and H(z) use
    Omega_m = Omega_DM + Omega_b, and the mass is capped (below).  In
    G = c = hbar = 1, 64 mu^6 M_c0 M_0^4 = 4 q |Gamma_0|.

    The bracket reaches zero at a finite time, where M -> infinity.  The mass
    is capped at M_0 + M_c0 (all the cloud absorbed), and z_depletion is the
    redshift at which the cap is reached.
    """

    def __init__(self, M_seed_Msun, M_cloud_Msun, mu_eV, z_seed, cosm):
        self.M_seed = float(M_seed_Msun)
        self.q = float(M_cloud_Msun) / self.M_seed
        self.mu_eV = float(mu_eV)
        self.cosm = cosm
        self.z_seed = float(z_seed)
        self.gamma0 = float(gamma_abs(self.M_seed, self.mu_eV))
        self.omega_L = 1.0 - cosm.omega_m_0
        # 2 K: the bracket is 1 - 2 K (F(z) - F(z_seed))
        self._rate = 4.0 * self.q * self.gamma0 * 2.0 \
            / (3.0 * cosm.hubble_0 * np.sqrt(self.omega_L))
        self.z_depletion = None

        self.absorbing = self.q > 0.0 and self.gamma0 > 0.0
        if not self.absorbing:
            return

        # Cap reached when the bracket equals (1 + q)^-4.
        x_cap = -np.expm1(-4.0 * np.log1p(self.q))
        F_cap = self.F(self.z_seed) + x_cap / self._rate
        y_cap = 1.0 / np.tanh(F_cap)     # inverse of arccoth
        if F_cap > 0.0:
            # (1 + z_cap)^3 = (y^2 - 1) Omega_L / Omega_m; z_cap >= 0 only
            z_cap = ((y_cap**2 - 1.0) * self.omega_L
                / cosm.omega_m_0)**(1.0 / 3.0) - 1.0
            if z_cap >= 0.0:
                self.z_depletion = float(z_cap)

    def F(self, z):
        """F(z) = arccoth sqrt(1 + a_1 (1 + z)^3); F(inf) = 0."""
        z = np.asarray(z, dtype=float)
        y2 = 1.0 + self.cosm.omega_m_0 / self.omega_L * (1.0 + z)**3
        return np.arctanh(1.0 / np.sqrt(y2))   # arccoth(y) = arctanh(1/y)

    def _mass_ratio(self, z):
        z = np.asarray(z, dtype=float)
        if not self.absorbing:
            return np.ones_like(z)
        Q = 1.0 + self.q
        x = self._rate * np.clip(self.F(z) - self.F(self.z_seed), 0.0, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            m = np.where(x < 1.0, (1.0 - x)**-0.25, Q)
        return np.minimum(m, Q)

    def M_cloud(self, z):
        """Cloud mass [Msun], from M_BH + M_c = M_0 + M_c0."""
        return _scalar_or_array(self.M_seed * (1.0 + self.q
            - self._mass_ratio(z)))

    def __call__(self, z):
        """PBH mass M_PBH(z) [Msun]."""
        return _scalar_or_array(self.M_seed * self._mass_ratio(z))


def _scalar_or_array(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x
