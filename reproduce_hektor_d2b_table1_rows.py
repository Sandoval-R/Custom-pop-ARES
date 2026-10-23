"""
Hektor + D2B runs for every case of the Dark2Black Table 1, at fixed beta.

Each row of Table 1 gives a scalar mass mu, a seed PBH mass M_BH,0 and an
initial cloud mass M_c,0.  For each row this script runs ARES with the
Hektor+D2B population (the cloud raises M_PBH(z), and so the Bondi
accretion) and, for comparison, the plain Hektor population with the same
seed mass and no cloud.

The D2B mass history is d2b_model.Mass_analytic, the closed-form
M_PBH(z) of the twentyoneSMBH draft (M_c held at M_c0, capped at
M_0 + M_c0), which plugs into the base model through mass_history.

Usage:
    python reproduce_hektor_d2b_table1_rows.py full        # ARES runs of all 11 rows
    python reproduce_hektor_d2b_table1_rows.py signal      # global 21-cm signal
    python reproduce_hektor_d2b_table1_rows.py masses  # M_PBH(z) plot, no ARES

full runs the PBHs alone (problem type 100) and plots Tk(z).  These runs
have no Lyman-alpha source, so Ts stays near T_CMB and their dTb is ~0.

signal runs the same rows as Full runs: ARES's standard galaxies (problem
type 101, default parameters, helium off) plus the PBHs, and a galaxies-only
reference.  It writes the global signal dTb(z) of every row together
(table1_signal.png), one figure per row with D2B on and off
(table1_signal_rowNN.png), and the depth and redshift of each absorption
trough (table1_signal.csv).

masses plots M_PBH(z) of every row and writes table1_mass.png.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import ares

from hektor_pbh_ares import make_pbh_population, DEFAULT_SED_PATH
from hektor_ares_patch import install_stateful_callbacks
from d2b_model import Mass_analytic


DEFAULT_COSM_KWARGS = dict(
    cosmology_name='planck_TTTEEE_lowl_lowE',
    cosmology_id='best',
)

# ============================================================================
# CONFIGURATION
# ============================================================================

BETA = 0.2            # Fixed for every run
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
# Full runs start lower: with ARES's galaxies (and their CGM phase) runs fail
# above z ~ 375.  300 is the start of the ares_git stage-2 Full runs.
Z_INITIAL_FULL = 300.0
Z_FINAL = 5.0
Z_REPORT = 17.2       # Redshift at which Tk is printed
Z_TROUGH_MAX = 40.0   # The absorption trough is searched for below this z
GALAXY_PROBLEM_TYPE = 101

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "table1_rows")

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

def galaxy_population_count():
    """Problem type 101's galaxy populations are tagged {0}..{n-1}; the PBHs
    become population n."""
    from ares.util.ProblemTypes import ProblemType
    ids = [int(key[key.rindex("{") + 1:-1])
           for key in ProblemType(GALAXY_PROBLEM_TYPE) if key.endswith("}")]
    return max(ids) + 1


def run_ares(pop_pars, full=False):
    """
    Run one history and return z, Tk, xe, dTb: the PBHs alone, or with
    full=True, a Full run (ARES's galaxies plus the PBHs).
    """
    z_initial = Z_INITIAL_FULL if full else Z_INITIAL

    # Switch the PBHs on just above the start and never off.
    pbh = dict(pop_pars)
    pbh["pop_zform"] = z_initial + 1.0
    pbh["pop_zdead"] = 0.0
    pbh.update(PBH_SOURCE_FLAGS)

    pars = dict(DEFAULT_COSM_KWARGS)
    pars["verbose"] = False
    pars["progress_bar"] = False
    if full:
        # The galaxies keep their own defaults, pop_zform (60) included.
        pop_id = galaxy_population_count()
        pars["problem_type"] = GALAXY_PROBLEM_TYPE
        pars.update({f"{key}{{{pop_id}}}": val for key, val in pbh.items()})
    else:
        pars["problem_type"] = 100
        pars.update(pbh)
    pars["first_light_redshift"] = z_initial + 1.0
    # Helium makes problem type 101's galaxies stall (ADR-0004 in ares_git).
    pars["include_He"] = not full

    pars["solver_atol"] = 1e-12
    pars["solver_rtol"] = 1e-10
    # ARES holds the rates fixed over a step; 0.01 Myr resolves a sudden
    # cloud depletion in cold gas (row 8: 0.1 Myr overshoots to 46 553 K,
    # 0.01 Myr gives 12 722.6 K; ticket 16 in ares_git).
    pars["max_timestep"] = 0.01

    pars["initial_redshift"] = z_initial
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


def d2b_history(mu_eV, M_seed, M_cloud):
    cosm = ares.physics.Cosmology(**DEFAULT_COSM_KWARGS)
    return Mass_analytic(M_seed, M_cloud, mu_eV, Z_SEED, cosm)


def run_case(M_seed, f_pbh=F_PBH, mass_history=None, full=False):
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
    return run_ares(pop_pars, full=full)


def Tk_at(z, Tk, z0=Z_REPORT):
    order = np.argsort(z)
    return float(np.interp(z0, z[order], Tk[order]))


def trough(z, dTb):
    """Depth [mK] and redshift of the absorption trough at z <= Z_TROUGH_MAX,
    or (nan, nan) if the signal is in emission throughout."""
    low = z <= Z_TROUGH_MAX
    i = np.argmin(dTb[low])
    if dTb[low][i] >= 0.0:
        return np.nan, np.nan
    return float(dTb[low][i]), float(z[low][i])


# ============================================================================
# MAIN
# ============================================================================

def plot_masses(rows):
    """M_PBH(z) / M_0 of every row, from the draft's expression."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))
    z = np.logspace(np.log10(Z_FINAL), np.log10(Z_INITIAL), 3000)
    print(f"{'row':>3} {'q':>8} {'z_dep':>10}")
    for color, i in zip(colors, rows):
        mu, M_seed, M_cloud, _ = TABLE1[i]
        history = d2b_history(mu, M_seed, M_cloud)
        ax.plot(z, history(z) / M_seed, color=color, lw=1.6, label=f"#{i}, "
                f"q = {M_cloud / M_seed:.3g}")
        z_dep = history.z_depletion
        z_dep = "none" if z_dep is None else f"{z_dep:.3f}"
        print(f"{i:>3} {M_cloud / M_seed:>8.4g} {z_dep:>10}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(Z_FINAL, Z_INITIAL)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$M_{\rm PBH}(z) / M_0$")
    ax.set_title("D2B mass history (twentyoneSMBH draft)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_mass.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def check_budget(rows):
    for i in rows:
        _, M_seed, M_cloud, _ = TABLE1[i]
        if F_PBH * (1.0 + M_cloud / M_seed) > 1.0:
            raise ValueError(f"row {i}: f_PBH (1 + M_c,0/M_BH,0) = "
                f"{F_PBH * (1.0 + M_cloud / M_seed):.3g} > 1 (ADR-0001)")


def main(rows):
    check_budget(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    install_stateful_callbacks()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))

    # LambdaCDM baseline: no PBHs.
    z0, Tk0, _, _ = run_case(M_seed=10.0, f_pbh=0.0)
    ax.plot(z0, Tk0, color="k", lw=2, label=r"$\Lambda$CDM")
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
        np.savetxt(os.path.join(OUTPUT_DIR, f"row{i:02d}_d2b.txt"),
                   np.column_stack([z, Tk, xe, dTb]),
                   header="z Tk[K] xe dTb[mK]")

        if M_seed not in hektor_cache:
            hektor_cache[M_seed] = run_case(M_seed)
        zh, Tkh, _, _ = hektor_cache[M_seed]
        rec["Tk_report_hektor_K"] = Tk_at(zh, Tkh)
        print(f"Tk(z={Z_REPORT}) Hektor = "
              f"{rec['Tk_report_hektor_K']:.4f} K")
        ax.plot(zh, Tkh, color=color, lw=0.9, ls="--", alpha=0.7)

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
    plt.close(fig)


def plot_row_signal(i, galaxies, d2b, hektor, z_dep):
    """
    One row's global signal: galaxies only, + Hektor (D2B off) and
    + Hektor + D2B (D2B on), each a pair (z, dTb), with the troughs marked
    and the depletion redshift z_dep (None if the cloud lasts past z = 0).
    """
    mu, M_seed, M_cloud, _ = TABLE1[i]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = [
        (galaxies, "galaxies only", "k", "-"),
        (hektor, "+ Hektor, D2B off", "tab:blue", "--"),
        (d2b, "+ Hektor + D2B, D2B on", "tab:red", "-"),
    ]
    for (z, dTb), name, color, ls in curves:
        depth, z_tr = trough(z, dTb)
        if np.isnan(depth):
            name += ": no trough"
        else:
            name += f": {depth:.1f} mK at z = {z_tr:.1f}"
            ax.plot(z_tr, depth, "o", color=color, ms=5)
        ax.plot(z, dTb, color=color, ls=ls, lw=1.6, label=name)

    if z_dep is not None and Z_FINAL < z_dep < 60.0:
        ax.axvline(z_dep, color="tab:red", lw=0.8, ls=":",
                   label=f"cloud depleted, z = {z_dep:.2f}")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlim(60, Z_FINAL)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\delta T_b$ [mK]")
    ax.set_title(fr"Row {i}: $\mu$ = {mu:.2e} eV, $M_0$ = {M_seed:.0e}, "
                 fr"$M_{{c,0}}$ = {M_cloud:.0e} $M_\odot$ "
                 fr"($q$ = {M_cloud / M_seed:.3g})"
                 + "\n" + fr"$\beta={BETA}$, $f_{{\rm PBH}}={F_PBH}$, "
                 fr"$\lambda={LAMBDA}$")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"table1_signal_row{i:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main_signal(rows):
    """Full runs (galaxies + PBHs) of every row: the global 21-cm signal."""
    check_budget(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    install_stateful_callbacks()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))

    # Galaxies only: no PBHs.
    zg, _, _, dTbg = run_case(M_seed=10.0, f_pbh=0.0, full=True)
    ax.plot(zg, dTbg, color="k", lw=2.2, label="galaxies only")
    depth, z_tr = trough(zg, dTbg)
    print(f"galaxies only: trough {depth:.1f} mK at z = {z_tr:.2f}")
    summary = [dict(row="galaxies", trough_d2b_mK=depth, z_trough_d2b=z_tr)]
    np.savetxt(os.path.join(OUTPUT_DIR, "signal_galaxies.txt"),
               np.column_stack([zg, dTbg]), header="z dTb[mK]")

    hektor_cache = {}  # Several rows share a seed mass

    for color, i in zip(colors, rows):
        mu, M_seed, M_cloud, _ = TABLE1[i]
        print("\n" + "=" * 72)
        print(f"row {i} (Full run): mu = {mu:.2e} eV, M_BH,0 = {M_seed:.1e} "
              f"Msun, M_c,0 = {M_cloud:.1e} Msun, beta = {BETA}")
        print("=" * 72)

        history = d2b_history(mu, M_seed, M_cloud)
        z, Tk, xe, dTb = run_case(M_seed, mass_history=history, full=True)
        z_dep = history.z_depletion
        depth, z_tr = trough(z, dTb)
        rec = dict(row=i, mu_eV=mu, M_seed_Msun=M_seed, M_cloud_Msun=M_cloud,
                   z_depletion=np.nan if z_dep is None else z_dep,
                   trough_d2b_mK=depth, z_trough_d2b=z_tr)
        print(f"depletion z = "
              f"{'none before z = 0' if z_dep is None else f'{z_dep:.3f}'}")
        print(f"D2B trough    = {depth:.1f} mK at z = {z_tr:.2f}")
        label = (fr"#{i}: $\mu$={mu:.1e}, $M_0$={M_seed:.0e}, "
                 fr"$M_c$={M_cloud:.0e}")
        ax.plot(z, dTb, color=color, lw=1.6, label=label)
        np.savetxt(os.path.join(OUTPUT_DIR, f"row{i:02d}_signal_d2b.txt"),
                   np.column_stack([z, Tk, xe, dTb]),
                   header="z Tk[K] xe dTb[mK]")

        if M_seed not in hektor_cache:
            hektor_cache[M_seed] = run_case(M_seed, full=True)
        zh, _, _, dTbh = hektor_cache[M_seed]
        rec["trough_hektor_mK"], rec["z_trough_hektor"] = trough(zh, dTbh)
        print(f"Hektor trough = {rec['trough_hektor_mK']:.1f} mK at "
              f"z = {rec['z_trough_hektor']:.2f}")
        ax.plot(zh, dTbh, color=color, lw=0.9, ls="--", alpha=0.7)

        plot_row_signal(i, (zg, dTbg), (z, dTb), (zh, dTbh), z_dep)

        summary.append(rec)

    keys = ["row", "mu_eV", "M_seed_Msun", "M_cloud_Msun", "z_depletion",
            "trough_d2b_mK", "z_trough_d2b", "trough_hektor_mK",
            "z_trough_hektor"]
    csv_path = os.path.join(OUTPUT_DIR, "table1_signal.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved: {csv_path}")

    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlim(60, Z_FINAL)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\delta T_b$ [mK]")
    ax.set_title(fr"Global 21-cm signal, galaxies + Hektor + D2B: "
                 fr"$\beta={BETA}$, $f_{{\rm PBH}}={F_PBH}$, "
                 fr"$\lambda={LAMBDA}$"
                 + "\n(dashed: Hektor, same seed, no cloud)")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "table1_signal.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["full", "signal", "masses"],
                        help="full: ARES runs of all rows, with the Hektor "
                             "comparison; signal: the same as Full runs "
                             "with ARES's galaxies, for the global 21-cm "
                             "signal; masses: M_PBH(z) of every row, "
                             "no ARES")
    args = parser.parse_args()

    if not os.path.exists(DEFAULT_SED_PATH):
        raise FileNotFoundError(f"Hektor SED not found: {DEFAULT_SED_PATH}")

    rows = list(range(len(TABLE1)))
    if args.mode == "masses":
        plot_masses(rows)
    elif args.mode == "signal":
        main_signal(rows)
    else:
        main(rows)
