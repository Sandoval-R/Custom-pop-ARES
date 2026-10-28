import os
import numpy as np
import matplotlib.pyplot as plt

import ares

from hektor_d2b_ares import (
    make_pbh_population,
    DEFAULT_SED_PATH,
)
from hektor_ares_patch import install_stateful_callbacks


DEFAULT_COSM_KWARGS = dict(
    cosmology_name='planck_TTTEEE_lowl_lowE',
    cosmology_id='best',
)

# ============================================================================
# CONFIGURATION
# ============================================================================

M_PBH = 1e2   # PBH seed mass [Msun]
F_PBH = 0.01          # PBH fraction of dark matter
LAMBDA = 0.01         # Hektor accretion parameter
F_E = 0.15            # Hektor deposited-energy fraction

BETAS = (0.2, 0.4, 0.6, 0.8, 1.0)

# ---------------------------------------------------------------------------
# D2B parameters
# ---------------------------------------------------------------------------

D2B_MU_EV = 1.73e-20
D2B_MC = 1.0e6       # initial scalar-cloud mass [Msun]

D2B_Z_START = 400.0


# ARES integration limits.
Z_INITIAL = D2B_Z_START
Z_FINAL = 5.0

# If True, also plot the original Hektor result as dashed curves.
# This makes the effect of adding D2B immediately visible.
COMPARE_WITH_HEKTOR = True

# Output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "figures")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "hektor_d2b_fig3_reproduction.png",
)
OUTPUT_GLOBAL_PATH = os.path.join(
    OUTPUT_DIR,
    "hektor_d2b_fig3_global_signal.png",
)


# ============================================================================
# ARES SETTINGS
# ============================================================================

PBH_SOURCE_FLAGS = dict(
    pop_lya_src=False,
    pop_ion_src_cgm=False,
    pop_ion_src_igm=True,
    pop_heat_src_cgm=False,
    pop_heat_src_igm=True,
    pop_solve_rte=False,
)


# ============================================================================
# D2B + ARES RUN
# ============================================================================

def run_d2b_case(beta):
    """
    Run one ARES history using the Hektor+D2B population.
    """

    pop_pars, model = make_pbh_population(
        M_PBH_Msun=M_PBH,
        f_PBH=F_PBH,
        lam=LAMBDA,
        beta=beta,
        f_E=F_E,
        sed_path=DEFAULT_SED_PATH,

        d2b_mu_eV=D2B_MU_EV,
        d2b_Mcloud_Msun=D2B_MC,
        d2b_z_start=D2B_Z_START,
        d2b_enabled=True,
    )

    pars = dict(DEFAULT_COSM_KWARGS)
    pars["problem_type"] = 100
    pars["verbose"] = False

    pars.update(pop_pars)

    # Override the factory defaults explicitly.
    pars["pop_zform"] = Z_INITIAL +1.0
    pars["pop_zdead"] = 0
    pars["first_light_redshift"] = Z_INITIAL + 1.0
    pars["include_He"] = True

    pars["solver_atol"] = 1e-12
    pars["solver_rtol"] = 1e-10
    pars["max_timestep"] = 0.1

    pars.update(PBH_SOURCE_FLAGS)

    pars.update(
        dict(
            initial_redshift=Z_INITIAL,
            final_redshift=Z_FINAL,
        )
    )

    sim = ares.simulations.Global21cm(**pars)
    sim.run()

    z = np.asarray(sim.history["z"])
    Tk = np.asarray(sim.history["igm_Tk"])
    dTb = np.asarray(sim.history["dTb"])

    if "igm_e" in sim.history:
        xe = np.asarray(sim.history["igm_e"])
    else:
        xe = 1.0 - np.asarray(sim.history["igm_h_1"])

    return z, Tk, xe, dTb, model


# ============================================================================
# ORIGINAL HEKTOR RUN -- OPTIONAL COMPARISON
# ============================================================================

def run_hektor_case(beta, f_pbh=F_PBH):
    """
    Run the original Hektor/ARES model, without D2B.
    """

    from hektor_pbh_ares import make_pbh_population as make_hektor_population

    pop_pars, model = make_hektor_population(
        M_PBH_Msun=M_PBH,
        f_PBH=f_pbh,
        lam=LAMBDA,
        beta=beta,
        f_E=F_E,
        sed_path=DEFAULT_SED_PATH,
    )

    pars = dict(DEFAULT_COSM_KWARGS)
    pars["problem_type"] = 100
    pars["verbose"] = False

    pars.update(pop_pars)

    pars["pop_zform"] = Z_INITIAL + 1.0
    pars["pop_zdead"] = 0
    pars["first_light_redshift"] = Z_INITIAL + 1.0
    pars["include_He"] = True

    pars["solver_atol"] = 1e-12
    pars["solver_rtol"] = 1e-10
    pars["max_timestep"] = 0.1

    pars.update(PBH_SOURCE_FLAGS)

    pars.update(
        dict(
            initial_redshift=Z_INITIAL,
            final_redshift=Z_FINAL,
        )
    )

    sim = ares.simulations.Global21cm(**pars)
    sim.run()

    z = np.asarray(sim.history["z"])
    Tk = np.asarray(sim.history["igm_Tk"])
    dTb = np.asarray(sim.history["dTb"])

    if "igm_e" in sim.history:
        xe = np.asarray(sim.history["igm_e"])
    else:
        xe = 1.0 - np.asarray(sim.history["igm_h_1"])

    return z, Tk, xe, dTb, model


# ============================================================================
# FIGURE 3
# ============================================================================

def figure3():
    """
    Generate the Hektor Figure-3-style Tk(z) plot with D2B accretion.

    The D2B curves are solid.  If COMPARE_WITH_HEKTOR=True, the original
    Hektor curves are dashed in the same color.
    """

    install_stateful_callbacks()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig_gs, ax_gs = plt.subplots(figsize=(6.5, 5.5))

    # ------------------------------------------------------------------
    # Lambda-CDM baseline
    # ------------------------------------------------------------------
    #
    # f_PBH=0 disables the PBH source. We can use the ordinary Hektor
    # factory because D2B is irrelevant when there is no PBH source.
    #
    z0, Tk0, xe0, dTb0, _ = run_hektor_case(beta=0.2, f_pbh=0.0)
    ax.plot(
        z0,
        Tk0,
        color="#4169E1",
        lw=2,
        label=r"$\Lambda$CDM",
    )
    ax_gs.plot(
        z0,
        dTb0,
        color="k",
        lw=2,
        label=r"$\Lambda$CDM",
    )

    # ------------------------------------------------------------------
    # Reference temperatures
    # ------------------------------------------------------------------
    z_cmb = np.logspace(
        np.log10(Z_FINAL),
        np.log10(1000.0),
        200,
    )

    ax.plot(
        z_cmb,
        2.725 * (1.0 + z_cmb),
        color="gray",
        ls="-",
        lw=1,
        label=r"$T_{\rm CMB}$",
    )

    beta_colors = {
        0.2: "#66C2A4",
        0.4: "#3AAE8C",
        0.6: "#39A852",
        0.8: "#4DBBD5",
        1.0: "#4169E1",
    }

    # ------------------------------------------------------------------
    # PBH curves
    # ------------------------------------------------------------------
    for beta in BETAS:

        print(
            "\n"
            + "=" * 72
            + f"\nbeta = {beta:.1f}"
            + "\n"
            + "=" * 72
        )

        # Hektor + D2B
        z, Tk, xe, dTb, model = run_d2b_case(beta)

        order = np.argsort(z)
        z_sorted = z[order]
        Tk_sorted = Tk[order]

        Tk_172 = np.interp(17.2, z_sorted, Tk_sorted)

        print(f"D2B cloud mass = {D2B_MC:.6e} Msun")
        print(f"D2B mu         = {D2B_MU_EV:.6e} eV")
        print(f"Tk(z=17.2)     = {Tk_172:.6f} K")

        if model.d2b is not None:
            if model.z_depletion is None:
                print("D2B cloud does not deplete between z=60 and z=6.")
            else:
                print(f"D2B depletion z = {model.z_depletion:.6f}")

            M_start = float(model.M_PBH(D2B_Z_START))
            M_end = float(model.M_PBH(Z_FINAL))

            print(f"M_BH(z=60)     = {M_start:.8e} Msun")
            print(f"M_BH(z=6)      = {M_end:.8e} Msun")

        ax.plot(
            z,
            Tk,
            color=beta_colors[beta],
            lw=1.8,
            label=fr"$\beta={beta:.1f}$, D2B",
        )

        ax_gs.plot(
            z,
            dTb,
            color=beta_colors[beta],
            lw=1.8,
            label=fr"$\beta={beta:.1f}$, D2B",
        )

        # --------------------------------------------------------------
        # Original Hektor comparison
        # --------------------------------------------------------------
        if COMPARE_WITH_HEKTOR:
            zh, Tkh, xeh, dTbh, _ = run_hektor_case(beta)

            ax.plot(
                zh,
                Tkh,
                color=beta_colors[beta],
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=fr"$\beta={beta:.1f}$, Hektor",
            )

            ax_gs.plot(
                zh,
                dTbh,
                color=beta_colors[beta],
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=fr"$\beta={beta:.1f}$, Hektor",
            )

    # ------------------------------------------------------------------
    # Hektor Figure-3-style annotations
    # ------------------------------------------------------------------
    ax.axvspan(
        15,
        21,
        color="gray",
        alpha=0.15,
        lw=0,
    )

    ax.axhline(
        7.0,
        color="C3",
        ls=":",
        lw=1,
    )
    ax.text(
        Z_INITIAL * 0.6,
        7.3,
        "7 K " + r"$\Lambda$CDM",
        fontsize=8,
        color="C3",
    )

    ax.axhline(
        4.0,
        color="C0",
        ls=":",
        lw=1,
    )
    ax.text(
        Z_INITIAL * 0.6,
        4.3,
        "4 K EDGES",
        fontsize=8,
        color="C0",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(10, 1000)
    ax.set_ylim(1, 2.725 * 1000)

    ax.set_xlabel("z")
    ax.set_ylabel(r"$T_k$ [K]")

    ax.set_title(
        fr"Hektor + D2B: $M_{{\rm PBH}}={M_PBH:.0f}\,M_\odot$, "
        fr"$\lambda={LAMBDA}$"
    )

    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"\nSaved: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Global 21-cm signal
    # ------------------------------------------------------------------
    ax_gs.set_xlabel("z")
    ax_gs.set_ylabel(r"$\delta T_b$ [mK]")

    ax_gs.set_title(
        fr"Hektor + D2B global 21-cm signal: "
        fr"$M_{{\rm PBH}}={M_PBH:.0f}\,M_\odot$, "
        fr"$\lambda={LAMBDA}$"
    )

    ax_gs.set_xlim(100, 5)
    ax_gs.legend(fontsize=7, ncol=2)

    fig_gs.tight_layout()
    fig_gs.savefig(
        OUTPUT_GLOBAL_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_GLOBAL_PATH}")

    plt.close(fig)
    plt.close(fig_gs)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    if not os.path.exists(DEFAULT_SED_PATH):
        raise FileNotFoundError(
            f"Hektor SED not found: {DEFAULT_SED_PATH}"
        )

    print("\nHektor + D2B Figure 3")
    print("---------------------")
    print(f"M_PBH       = {M_PBH} Msun")
    print(f"f_PBH       = {F_PBH}")
    print(f"lambda      = {LAMBDA}")
    print(f"f_E         = {F_E}")
    print(f"betas       = {BETAS}")
    print(f"D2B mu      = {D2B_MU_EV:.6e} eV")
    print(f"D2B M_cloud = {D2B_MC:.6e} Msun")
    print(f"z range     = {Z_INITIAL} -> {Z_FINAL}")
    print()

    figure3()
