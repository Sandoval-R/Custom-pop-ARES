"""
---
The model is
    |Gamma| = 16 (mu M_BH)^6 / M_BH
with
    dM_BH/dt    = |Gamma| M_cloud
    dM_cloud/dt = -|Gamma| M_cloud.
Masses are in solar masses and time is in years.
---
Considering that
    M_BH + M_cloud = M_total
is conserved, the coupled system can be reduced exactly to one equation by defining
    u = ln(M_cloud,0 / M_cloud).
Then
    M_cloud = M_cloud,0 exp(-u),
    M_BH    = M_total - M_cloud,
and
    du/dt = Gamma(M_BH).

This variable is numerically much better behaved near cloud depletion
than integrating M_cloud directly: the cloud never has to be divided
by zero, and the depletion event is simply u = u_final.
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


class DarkToBlack:
    """
    Evolve one black hole embedded in a finite scalar cloud.
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
        method="RK45", # ODE solver, is Runge-Kutta
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

        if self.method not in ("Radau", "BDF", "LSODA", "RK45", "DOP853"):
            raise ValueError(
                "Unsupported method. Use Radau, BDF, LSODA, RK45, or DOP853."
            )

        self.M_total = self.Mbh0 + self.Mc0 # Total mass of the system

        # u at the numerical cloud-depletion threshold.
        self.u_final = np.log(self.Mc0 / self.cloud_floor)

        self._solution = None
        self.t_depletion_yr = None

    # ------------------------------------------------------------------
    # alpha = mu*M in natural units
    # ------------------------------------------------------------------

    def alpha(self, Mbh):
        """
        Dimensionless gravitational coupling alpha = mu M_BH.

        In the paper mu[eV]*M_BH[m] -> dimensionless
        So we need to recover units before that multiplication
        mu is supplied as an energy, converted to inverse metres:

            mu [1/m] = mu[eV] eV[J] / (hbar c)

        and the BH mass is converted to gravitational length:

            M = G M_BH / c^2.
        """
        mu_inv_m = self.mu_eV * EV_J / (HBAR * c)
        M_geom_m = G * np.asarray(Mbh) * M_SUN / c**2 # To allow multiple masses
        return mu_inv_m * M_geom_m

    # ------------------------------------------------------------------
    # Dark-to-Black rate
    # ------------------------------------------------------------------

    def gamma_abs(self, Mbh):
        """
        Return |Gamma| in s^-1.
        In natural units (G = c = hbar = 1),
            |Gamma| = 16 alpha^6 / M
        where alpha = mu M is dimensionless.

        Restoring units,
            |Gamma| = 16 alpha^6 c^3 / (G M_BH).

        The corresponding cloud e-folding time is
            t_E = |Gamma|^-1.
        """
        M_kg = np.asarray(Mbh) * M_SUN
        return 16.0 * self.alpha(Mbh)**6 * c**3 / (G * M_kg)

    def mdot(self, Mbh, Mc):
        """
        Instantaneous Dark-to-Black accretion rate in Msun/yr.
            dM_BH/dt = |Gamma| M_cloud, [1/s]
        And we multiply the YEAR to give proper units
        """
        return self.gamma_abs(Mbh) * np.asarray(Mc) * YEAR

    def t_efold_yr(self, Mbh):
        """Scalar-cloud e-folding time in years.

            t_E = |Gamma| [s],

            so divide by YEAR to get units
        """
        return 1.0 / self.gamma_abs(Mbh) / YEAR

    # ------------------------------------------------------------------
    # Reduced ODE
    # ------------------------------------------------------------------

    def _rhs_u(self, t_yr, y):
        """
        In the model
            M_total = M_BH + M_c,
        and
            dM_BH/dt = |Gamma| M_cloud,
            dM_c/dt = -|Gamma| M_cloud.
        So we can reduce the equation to solve just one, let
            u = ln(Mc0 / Mc)
        such that
            e^-u = Mc / Mc0 ---> Mc = Mc0 e^-u.
        Then
            du/dt = -1/Mc dMc/dt
                  = (-1/Mc) (-|Gamma|Mc)
                  = |Gamma|
        and
            M_BH = M_total - Mc0 exp(-u).
        Solving directly had an error so I tried this to get a better behaved function. Didn't work
        """
        u = max(float(y[0]), 0.0) # u must be positive for depleting cloud

        Mc = self.Mc0 * np.exp(-u)
        Mbh = self.M_total - Mc

        gamma = float(self.gamma_abs(Mbh))

        # Gamma is in s^-1, while t is measured in years.
        return np.array([gamma * YEAR])

    def _depletion_event_u(self, t_yr, y):
        """Terminate at the numerical cloud-depletion threshold."""
        return float(y[0]) - self.u_final

    _depletion_event_u.terminal = True
    _depletion_event_u.direction = 1.0

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, n_output=10000):
        """
        Solve the D2B evolution.

        Internally this solves only the reduced u-equation. M_BH and
        M_cloud are reconstructed exactly from mass conservation.

        If the cloud depletes before t_end_yr, the evolution is stopped
        at depletion and the remaining requested output times are filled
        with the final BH mass, zero cloud mass (to the numerical floor),
        and zero D2B accretion rate.
        """

        t_eval = np.linspace(
            self.t0_yr,
            self.t_end_yr,
            int(n_output),
        )

        u0 = np.array([0.0], dtype=float)

        sol = solve_ivp(
            self._rhs_u,
            (self.t0_yr, self.t_end_yr),
            u0,
            t_eval = None,
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

        if len(sol.t_events[0]):
            self.t_depletion_yr = float(sol.t_events[0][0])
        else:
            self.t_depletion_yr = None

        # Evaluate u on the requested output grid, but only while the
        # physical cloud is above the depletion threshold.
        if self.t_depletion_yr is None:
            active = np.ones_like(t_eval, dtype=bool)
            u = np.asarray(sol.sol(t_eval)[0], dtype=float)
        else:
            active = t_eval < self.t_depletion_yr
            u = np.empty_like(t_eval)

            if np.any(active):
                u[active] = sol.sol(t_eval[active])[0]

            # At and after depletion, hold the final state fixed.
            u[~active] = self.u_final

        # Guard against tiny numerical excursions beyond the event.
        u = np.clip(u, 0.0, self.u_final)

        # Reconstruct both masses exactly from the conserved total mass.
        Mc = self.Mc0 * np.exp(-u)
        Mbh = self.M_total - Mc

        # The numerical floor is treated as complete depletion for
        # the purpose of the luminosity/accretion source term.
        depleted = np.zeros_like(t_eval, dtype=bool)
        if self.t_depletion_yr is not None:
            depleted = t_eval >= self.t_depletion_yr

        mdot = self.mdot(Mbh, Mc)
        mdot = np.where(depleted, 0.0, mdot)

        # Report the cloud as zero after depletion, while retaining the
        # exact pre-depletion numerical floor internally.
        Mc_output = np.where(depleted, 0.0, Mc)

        return {
            "t_yr": t_eval,
            "Mbh": Mbh,
            "Mc": Mc_output,
            "mdot": mdot,
            "alpha": self.alpha(Mbh),
            "t_efold_yr": self.t_efold_yr(Mbh),
            "u": u,
            "t_depletion_yr": self.t_depletion_yr,
        }
    def state_at(self, t_yr):
        """Evaluate the solved D2B state at arbitrary cosmic time(s).
        Useful for the ARES integration
        """

        t = np.asarray(t_yr, dtype=float)
        u = np.asarray(self._solution.sol(t)[0], dtype=float)

        if self.t_depletion_yr is not None:
            depleted = t >= self.t_depletion_yr
            u = np.where(depleted, self.u_final, u)
        else:
            depleted = np.zeros_like(t, dtype=bool)

        u = np.clip(u, 0.0, self.u_final)

        Mc_internal = self.Mc0 * np.exp(-u)
        Mbh = self.M_total - Mc_internal
        mdot = self.mdot(Mbh, Mc_internal)
        mdot = np.where(depleted, 0.0, mdot)
        Mc = np.where(depleted, 0.0, Mc_internal)

        return {
            "t_yr": t,
            "Mbh": Mbh,
            "Mc": Mc,
            "mdot": mdot,
            "alpha": self.alpha(Mbh),
            "t_efold_yr": self.t_efold_yr(Mbh),
            "u": u,
            "t_depletion_yr": self.t_depletion_yr,
        }
