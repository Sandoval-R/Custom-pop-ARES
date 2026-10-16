"""
D2B mass history for ARES_adapter, solved in closed form so that it works
for any cloud ratio.

    dM_BH/dt = |Gamma| M_c,   dM_c/dt = -|Gamma| M_c,
    |Gamma|  = 16 alpha^6 c^3 / (G M_BH),   alpha = mu G M_BH / (hbar c^3).

M_BH + M_c is conserved, so the system reduces to one equation in
u = ln(M_c0 / M_c).  Since |Gamma| scales as M_BH^5, measuring time in
units of the seed's e-folding time, s = |Gamma_0| (t - t_seed), gives

    du/ds = m^5,   m = M_BH / M_BH,0 = 1 + q (1 - e^-u),   q = M_c0 / M_BH,0.

Its inverse, ds/du = m^-5, integrates exactly (partial fractions in m, with
Q = 1 + q and Q - m = q e^-u):

    s(u) = Q^-5 (ln m + u) + sum_{k=2..5} Q^-(6-k) (1 - m^(1-k)) / (k-1),

and u(s) is found by root finding.  Any ODE solver fails once q is large:
near depletion du/ds = Q^5, so for q ~ 1e4 the last e-folds of the cloud
take ~1e-20 of an e-folding time, below the float spacing of s.  That is
why d2b_model.DarkToBlack (cosmic time, RK45) and ares_git's ticket-06
solver (s, Radau) both stop on Table 1 rows 0 and 9 with "Required step
size is less than spacing between numbers".

An instance is a function M_PBH(z) [Msun], so it plugs into
hektor_pbh_ares.make_pbh_population(mass_history=...) directly.
"""
import numpy as np
from scipy.optimize import brentq

from hektor_pbh_ares import MSUN_G

G_CGS = 6.67430e-8              # Gravitational constant [cm^3/g/s^2]
C_CGS = 2.99792458e10           # Speed of light [cm/s]
HBAR_CGS = 1.054571817e-27      # Reduced Planck constant [erg s]
ERG_PER_EV = 1.602176634e-12

# The cloud is used up once M_c < DEPLETED_FRACTION M_c0.  Past that point
# the cloud decays at the constant rate of the final BH, so u continues
# analytically.
DEPLETED_FRACTION = 1e-9
U_DEPLETED = -np.log(DEPLETED_FRACTION)


def cosmic_time(cosm, z):
    """
    Cosmic time t(z) [s] of ARES's matter + lambda universe
    (Cosmology.t_of_z), written with asinh: ARES's log form loses ~1e-9 of
    t at early times, which the runaway regime amplifies.
    """
    x = (1.0 / (1.0 + np.asarray(z, dtype=float)) / cosm.a_eq)**1.5
    return np.arcsinh(x) / _expansion_rate(cosm)


def redshift_at(cosm, t):
    """Inverse of cosmic_time (ARES's Cosmology.z_of_t)."""
    a = cosm.a_eq * np.sinh(_expansion_rate(cosm) * np.asarray(t))**(2.0 / 3.0)
    return 1.0 / a - 1.0


def _expansion_rate(cosm):
    return 1.5 * np.sqrt(1.0 - cosm.omega_m_0) * cosm.hubble_0


def alpha(M_Msun, mu_eV):
    """Gravitational coupling alpha = mu G M / (hbar c^3)."""
    return (mu_eV * ERG_PER_EV) * G_CGS * (M_Msun * MSUN_G) \
        / (HBAR_CGS * C_CGS**3)


def gamma_abs(M_Msun, mu_eV):
    """Absorption rate |Gamma| = 16 alpha^6 c^3 / (G M) [1/s]."""
    return 16.0 * alpha(M_Msun, mu_eV)**6 * C_CGS**3 \
        / (G_CGS * M_Msun * MSUN_G)


class D2BMassHistory:
    """
    M_PBH(z) [Msun] of a seed of mass M_BH,0 carrying a cloud M_c0,
    absorbing it from z_seed on (z_seed = inf is cosmic time t = 0).  Before
    z_seed the mass is the seed mass.

    z_depletion is the redshift at which the cloud is used up, or None if it
    survives to z = 0.
    """

    def __init__(self, M_seed_Msun, M_cloud_Msun, mu_eV, z_seed, cosm):
        if M_seed_Msun <= 0:
            raise ValueError('M_seed_Msun must be > 0.')
        if M_cloud_Msun < 0 or mu_eV < 0:
            raise ValueError('M_cloud_Msun and mu_eV must be >= 0.')

        self.M_seed = float(M_seed_Msun)
        self.q = float(M_cloud_Msun) / self.M_seed
        self.mu_eV = float(mu_eV)
        self.cosm = cosm
        self.t_seed = float(cosmic_time(cosm, z_seed))
        self.gamma0 = float(gamma_abs(self.M_seed, self.mu_eV))
        self.z_depletion = None

        self.absorbing = self.q > 0.0 and self.gamma0 > 0.0
        if not self.absorbing:
            return

        self._s_depletion = self.s_of_u(U_DEPLETED)
        t_depletion = self.t_seed + self._s_depletion / self.gamma0
        if t_depletion < float(cosmic_time(cosm, 0.0)):
            self.z_depletion = float(redshift_at(cosm, t_depletion))

    def _mass_ratio(self, u):
        """M_BH / M_BH,0 at u = ln(M_c0 / M_c)."""
        return 1.0 + self.q * -np.expm1(-u)

    def s_of_u(self, u):
        """Seed e-folding times s until the cloud reaches u (exact)."""
        Q = 1.0 + self.q
        log_m = np.log1p(self.q * -np.expm1(-u))
        s = Q**-5 * (log_m + u)
        for k in range(2, 6):
            s += Q**-(6 - k) * -np.expm1((1 - k) * log_m) / (k - 1)
        return s

    def _u_of_s(self, s):
        if s <= 0.0:
            return 0.0
        if s >= self._s_depletion:
            # The cloud decays at the constant rate of the final BH.
            return U_DEPLETED + (1.0 + self.q)**5 * (s - self._s_depletion)
        return brentq(lambda u: self.s_of_u(u) - s, 0.0, U_DEPLETED,
            xtol=1e-14, rtol=1e-14)

    def u(self, z):
        """u = ln(M_c0 / M_c) at redshift z."""
        z = np.asarray(z, dtype=float)
        if not self.absorbing:
            return np.zeros_like(z)
        s = self.gamma0 * (np.atleast_1d(cosmic_time(self.cosm, z))
            - self.t_seed)
        u = np.array([self._u_of_s(si) for si in s])
        return u.reshape(z.shape)

    def M_cloud(self, z):
        """Cloud mass [Msun]."""
        return _scalar_or_array(self.q * self.M_seed * np.exp(-self.u(z)))

    def __call__(self, z):
        """PBH mass M_PBH(z) [Msun]."""
        return _scalar_or_array(self.M_seed * self._mass_ratio(self.u(z)))


class DraftEq9MassHistory:
    """
    M_PBH(z) [Msun] from Eq. (9) of the draft, for comparison
    with D2BMassHistory.  Same arguments and interface.

    Eq. (8) integrates Eq. (7) with M_c taken out of the integral, i.e. at
    its initial value M_c0:

        M(z) = M_0 [1 - 128 mu^6 M_c0 M_0^4 (F(z) - F(z_seed))
                        / (3 H_0 sqrt(Omega_L))]^(-1/4),
        F(z) = arccoth sqrt(1 + (Omega_m / Omega_L) (1 + z)^3).

    Written as in the draft except for three corrections: M_0^4 rather than
    M_0 in the denominator (dimensions), arccoth rather than arctanh (the
    argument is > 1 and they both have the same derivate), and
    Omega_m = Omega_DM + Omega_b in H(z).
    In G = c = hbar = 1, 64 mu^6 M_c0 M_0^4 = 4 q |Gamma_0|, which is what the code uses.

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
        # 128 mu^6 M_c0 M_0^4 / (3 H_0 sqrt(Omega_L)), in units of F
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
        """Eq. (10) with arccoth; F(inf) = 0."""
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
