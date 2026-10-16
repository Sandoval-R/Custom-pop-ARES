"""
Hektor + D2B runs for every case of the Dark2Black Table 1, at fixed beta.

Each row of Table 1 gives a scalar mass mu, a seed PBH mass M_BH,0 and an
initial cloud mass M_c,0.  For each row this script runs ARES with the
Hektor+D2B population (the cloud raises M_PBH(z), and so the Bondi
accretion) and, for comparison, the plain Hektor population with the same
seed mass and no cloud.

The D2B mass history comes from d2b_mass_history.py (closed form), not from
hektor_d2b_ares.py: d2b_model.DarkToBlack stops on rows 0 and 9 with
"Required step size is less than spacing between numbers".  It plugs into
the base model through mass_history.

Usage:
    python reproduce_hektor_d2b_table1.py full        # ARES runs of all 11 rows
    python reproduce_hektor_d2b_table1.py masses  # M_PBH(z) plot, no ARES

masses compares M_PBH(z) of the exact history with Eq. (9) of the
twentyoneSMBH draft (DraftEq9MassHistory: M_c held at M_c0 in the
integral, capped at M_0 + M_c0) and writes table1_mass.png.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import ares

from hektor_pbh_ares import make_pbh_population, DEFAULT_SED_PATH
from hektor_ares_patch import install_stateful_callbacks
from d2b_mass_history import D2BMassHistory, DraftEq9MassHistory


DEFAULT_COSM_KWARGS = dict(
    cosmology_name='planck_TTTEEE_lowl_lowE',
    cosmology_id='best',
)

# ============================================================================
# CONFIGURATION
# ============================================================================

BETA = 0.6            # Fixed for every run
# Seed fraction of dark matter, the same for every row.
F_PBH = 1e-5
LAMBDA = 0.01         # Hektor accretion parameter
F_E = 0.15            # Hektor deposited-energy fraction

# Dark2Black Table 1: (mu [eV], M_BH,0 [Msun], M_c,0 [Msun], t_dep [yr]).
# t_dep is the paper's depletion time, written to the summary for reference.
TABLE1 = [
    (1.00e-16, 1e2, 1e6, 1.4e8),
    (1.73e-17, 5e2, 1e6, 8.3e9),
    (1.73e-17, 1e3, 1e6, 5.2e8),
    (5.00e-18, 5e3, 1e7, 1.4e8),
    (7.00e-19, 5e3, 1e8, 1.9e12),
    (7.00e-19, 5e4, 1e8, 1.9e8),
    (5.00e-19, 1e5, 1e7, 8.9e8),
    (5.00e-19, 6e4, 1e8, 6.9e8),
    (5.00e-19, 1e5, 1e8, 8.9e7),
    (3.00e-19, 8e4, 1e9, 4.7e8),
    (9.00e-20, 5e5, 1e9, 4.2e8),
]

Z_SEED = 1090.0       # Clouds start being absorbed at recombination (ADR-0003)
Z_INITIAL = 400.0
Z_FINAL = 5.0
Z_REPORT = 17.2       # Redshift at which Tk is printed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "table1")

PBH_SOURCE_FLAGS = dict(
    pop_lya_src=False,
    pop_ion_src_cgm=False,
    pop_ion_src_igm=True,
    pop_heat_src_cgm=False,
    pop_heat_src_igm=True,
    pop_solve_rte=False,
)


# ============================================================================
# ARES RUN
# ============================================================================

def run_ares(pop_pars):
    """Run one PBH-only history and return z, Tk, xe, dTb."""
    pars = dict(DEFAULT_COSM_KWARGS)
    pars["problem_type"] = 100
    pars["verbose"] = False
    pars.update(pop_pars)

    # Switch the PBHs on just above the start and never off.
    pars["pop_zform"] = Z_INITIAL + 1.0
    pars["pop_zdead"] = 0.0
    pars["first_light_redshift"] = Z_INITIAL + 1.0
    pars["include_He"] = True

    pars["solver_atol"] = 1e-12
    pars["solver_rtol"] = 1e-10
    # ARES holds the rates fixed over a step; 0.1 Myr resolves a sudden
    # cloud depletion in cold gas (ticket 16 in ares_git).
    pars["max_timestep"] = 0.1

    pars.update(PBH_SOURCE_FLAGS)
    pars["initial_redshift"] = Z_INITIAL
    pars["final_redshift"] = Z_FINAL

    sim = ares.simulations.Global21cm(**pars)
    if not sim.medium.field.solver.approx_all_pops:
        raise RuntimeError('A population solves the RTE: ARES would tabulate '
            'the PBH callbacks once and freeze them.')
    sim.run()

    z = np.asarray(sim.history["z"])
    Tk = np.asarray(sim.history["igm_Tk"])
    dTb = np.asarray(sim.history["dTb"])
    xe = 1.0 - np.asarray(sim.history["igm_h_1"])
    return z, Tk, xe, dTb


def d2b_history(mu_eV, M_seed, M_cloud, model=D2BMassHistory):
    cosm = ares.physics.Cosmology(**DEFAULT_COSM_KWARGS)
    return model(M_seed, M_cloud, mu_eV, Z_SEED, cosm)


def run_case(M_seed, f_pbh=F_PBH, mass_history=None):
    """Hektor's model; with a mass_history, Hektor + D2B."""
    pop_pars, model = make_pbh_population(
        M_PBH_Msun=M_seed,
        f_PBH=f_pbh,
        lam=LAMBDA,
        beta=BETA,
        f_E=F_E,
        sed_path=DEFAULT_SED_PATH,
        mass_history=mass_history,
    )
    return run_ares(pop_pars)


def Tk_at(z, Tk, z0=Z_REPORT):
    order = np.argsort(z)
    return float(np.interp(z0, z[order], Tk[order]))


# ============================================================================
# MAIN
# ============================================================================

def plot_masses(rows):
    """M_PBH(z) / M_0 of the exact history (solid) and Eq. (9) (dotted)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))
    z = np.logspace(np.log10(Z_FINAL), np.log10(Z_INITIAL), 3000)
    print(f"{'row':>3} {'q':>8} {'z_dep exact':>12} {'z_dep Eq9':>10}")
    for color, i in zip(colors, rows):
        mu, M_seed, M_cloud, _ = TABLE1[i]
        exact = d2b_history(mu, M_seed, M_cloud)
        eq9 = d2b_history(mu, M_seed, M_cloud, DraftEq9MassHistory)
        ax.plot(z, exact(z) / M_seed, color=color, lw=1.6, label=f"#{i}, "
                f"q = {M_cloud / M_seed:.3g}")
        ax.plot(z, eq9(z) / M_seed, color=color, lw=1.6, ls=":")
        fmt = lambda zd: "none" if zd is None else f"{zd:.3f}"
        print(f"{i:>3} {M_cloud / M_seed:>8.4g} {fmt(exact.z_depletion):>12} "
              f"{fmt(eq9.z_depletion):>10}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(Z_FINAL, Z_INITIAL)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$M_{\rm PBH}(z) / M_0$")
    ax.set_title("D2B mass history: exact (solid) vs draft Eq. (9) (dotted)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_mass.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main(rows):
    for i in rows:
        _, M_seed, M_cloud, _ = TABLE1[i]
        if F_PBH * (1.0 + M_cloud / M_seed) > 1.0:
            raise ValueError(f"row {i}: f_PBH (1 + M_c,0/M_BH,0) = "
                f"{F_PBH * (1.0 + M_cloud / M_seed):.3g} > 1 (ADR-0001)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    install_stateful_callbacks()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig_gs, ax_gs = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))

    # LambdaCDM baseline: no PBHs.
    z0, Tk0, _, dTb0 = run_case(M_seed=10.0, f_pbh=0.0)
    ax.plot(z0, Tk0, color="k", lw=2, label=r"$\Lambda$CDM")
    ax_gs.plot(z0, dTb0, color="k", lw=2, label=r"$\Lambda$CDM")
    summary = [dict(row="LCDM", Tk_report_d2b_K=Tk_at(z0, Tk0))]

    hektor_cache = {}  # Several rows share a seed mass

    for color, i in zip(colors, rows):
        mu, M_seed, M_cloud, t_dep_paper = TABLE1[i]
        print("\n" + "=" * 72)
        print(f"row {i}: mu = {mu:.2e} eV, M_BH,0 = {M_seed:.1e} Msun, "
              f"M_c,0 = {M_cloud:.1e} Msun, beta = {BETA}")
        print("=" * 72)

        history = d2b_history(mu, M_seed, M_cloud)
        z, Tk, xe, dTb = run_case(M_seed, mass_history=history)
        M_end = history(Z_FINAL)
        z_dep = history.z_depletion
        rec = dict(
            row=i, mu_eV=mu, M_seed_Msun=M_seed, M_cloud_Msun=M_cloud,
            t_dep_paper_yr=t_dep_paper,
            z_depletion=np.nan if z_dep is None else z_dep,
            M_end_Msun=M_end,
            Tk_report_d2b_K=Tk_at(z, Tk),
        )
        print(f"depletion z    = "
              f"{'none before z = 0' if z_dep is None else f'{z_dep:.3f}'}")
        print(f"M_BH(z={Z_FINAL:g})    = {M_end:.4e} Msun")
        print(f"Tk(z={Z_REPORT}) D2B = {rec['Tk_report_d2b_K']:.4f} K")

        label = (fr"#{i}: $\mu$={mu:.1e}, $M_0$={M_seed:.0e}, "
                 fr"$M_c$={M_cloud:.0e}")
        ax.plot(z, Tk, color=color, lw=1.6, label=label)
        ax_gs.plot(z, dTb, color=color, lw=1.6, label=label)
        np.savetxt(os.path.join(OUTPUT_DIR, f"row{i:02d}_d2b.txt"),
                   np.column_stack([z, Tk, xe, dTb]),
                   header="z Tk[K] xe dTb[mK]")

        if M_seed not in hektor_cache:
            hektor_cache[M_seed] = run_case(M_seed)
        zh, Tkh, _, dTbh = hektor_cache[M_seed]
        rec["Tk_report_hektor_K"] = Tk_at(zh, Tkh)
        print(f"Tk(z={Z_REPORT}) Hektor = "
              f"{rec['Tk_report_hektor_K']:.4f} K")
        ax.plot(zh, Tkh, color=color, lw=0.9, ls="--", alpha=0.7)
        ax_gs.plot(zh, dTbh, color=color, lw=0.9, ls="--", alpha=0.7)

        summary.append(rec)

    # Write the summary table.
    keys = ["row", "mu_eV", "M_seed_Msun", "M_cloud_Msun", "t_dep_paper_yr",
            "z_depletion", "M_end_Msun", "Tk_report_d2b_K",
            "Tk_report_hektor_K"]
    csv_path = os.path.join(OUTPUT_DIR, "table1_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved: {csv_path}")

    # Tk(z)
    z_cmb = np.logspace(np.log10(Z_FINAL), np.log10(1000.0), 200)
    ax.plot(z_cmb, 2.725 * (1.0 + z_cmb), color="gray", lw=1,
            label=r"$T_{\rm CMB}$")
    ax.axvspan(15, 21, color="gray", alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(Z_FINAL, Z_INITIAL)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$T_k$ [K]")
    ax.set_title(fr"Hektor + D2B, Table 1 cases: $\beta={BETA}$, "
                 fr"$f_{{\rm PBH}}={F_PBH}$, $\lambda={LAMBDA}$"
                 + "\n(dashed: Hektor, same seed, no cloud)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_Tk.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # dTb(z)
    ax_gs.set_xlim(100, Z_FINAL)
    ax_gs.set_xlabel("z")
    ax_gs.set_ylabel(r"$\delta T_b$ [mK]")
    ax_gs.set_title(fr"Hektor + D2B global signal: $\beta={BETA}$")
    ax_gs.legend(fontsize=6)
    fig_gs.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_dTb.png")
    fig_gs.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.close(fig)
    plt.close(fig_gs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["full", "masses"],
                        help="full: ARES runs of all rows, with the Hektor "
                             "comparison; masses: M_PBH(z) of the exact "
                             "history and Eq. (9), no ARES")
    args = parser.parse_args()

    if not os.path.exists(DEFAULT_SED_PATH):
        raise FileNotFoundError(f"Hektor SED not found: {DEFAULT_SED_PATH}")

    rows = list(range(len(TABLE1)))
    if args.mode == "masses":
        plot_masses(rows)
    else:
        main(rows)
