"""
Reproduce Hektor/ARES results, in a clean or a full universe.

Joins reproduce_hektor.py (clean) and reproduce_hektor_full.py (full):
    clean : problem_type 100, ONLY the Hektor PBH source, helium on, z_i = 400
    full  : ProblemType(101), ARES's own galaxy populations plus the PBH
            source as an extra population, helium off, z_i = 400

Both use the plain Hektor accretion model (hektor_pbh_ares), no D2B.

Usage:
    python reproduce_hektor_fig3.py clean
    python reproduce_hektor_fig3.py full
    python reproduce_hektor_fig3.py both
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from hektor_pbh_ares import (
    make_pbh_population,
    DEFAULT_COSM_KWARGS,
    DEFAULT_SED_PATH,
)

from hektor_ares_patch import install_stateful_callbacks

import ares
from ares.util.ProblemTypes import ProblemType

# Explicitly install the Hektor-specific ARES callback patch.
install_stateful_callbacks()

PBH_SOURCE_FLAGS = dict(
    pop_lya_src=False,
    pop_ion_src_cgm=False,
    pop_ion_src_igm=True,
    pop_heat_src_cgm=False,
    pop_heat_src_igm=True,
    pop_solve_rte=False,
)

# Everything that differs between the two universes lives here.
UNIVERSES = {
    'clean': dict(
        z_initial=400,  # ARES can handle up to 400 without problems
        include_He=True,
        file_prefix='hektor',
    ),
    'full': dict(
        z_initial=400,
        include_He=False,  # For full runs it doesn't work properly :c
        file_prefix='hektor_full',
    ),
}

Z_FINAL = 6.0


# ---------------------------------------------------------------------------
# Add custom population
# ---------------------------------------------------------------------------
def add_pbh_population(base_pars, pop_id, pop_pars, z_start=None):
    """
    Bolt the Hektor PBH source on as an ADDITIONAL population in an already-existing multi-population ARES run, instead of replacing whatever `base_pars` already sets up. Useful for the full run.

    base_pars : dict
        Your existing run's full parameter dict, with its populations already tagged {0}, {1}, {2}, ... (whatever ARES convention you're already using). Left completely untouched/copied, not mutated.
    pop_id : int
        The ID number for the new PBH population.
    pop_pars : dict
        The untagged PBH population parameters, from make_pbh_population.

    Returns a NEW dict: base_pars plus the PBH population's parameters,
    each tagged '{pop_id}', ready to pass to ares.simulations.Global21cm.

    """
    if z_start is None:
        z_start = base_pars['initial_redshift']
    pop_pars = dict(pop_pars, pop_zform=z_start + 1.0, pop_zdead=0.0)

    merged = dict(base_pars)  # copy, don't mutate caller's dict
    for key, val in pop_pars.items():
        merged[f'{key}{{{pop_id}}}'] = val
    for key, val in PBH_SOURCE_FLAGS.items():
        merged[f'{key}{{{pop_id}}}'] = val

    merged['first_light_redshift'] = z_start + 1.0

    return merged


# ---------------------------------------------------------------------------
# Parameter sets for each universe
# ---------------------------------------------------------------------------
def clean_pars(pop_pars, z_initial):
    """Only the Hektor PBH source, on problem_type 100."""
    pars = dict(DEFAULT_COSM_KWARGS)
    pars['problem_type'] = 100 # Clean universe
    pars.update(pop_pars)
    pars['pop_zform'] = z_initial + 1.0
    pars['first_light_redshift'] = z_initial + 1.0
    pars['pop_zdead'] = 0.0
    pars.update(PBH_SOURCE_FLAGS)
    return pars


def full_pars(pop_pars, z_initial, pop_id=None):
    """ARES's default run (3 galaxy populations) plus the PBH source."""
    base_pars = ProblemType(101) # Default ARES run, comes with 3 populations

    # -- Cleaner way to add population instead of replace --
    if pop_id is None:
        # The next id after ARES's own galaxy populations, read off the tags.
        pop_id = 1 + max(int(key[key.rindex('{') + 1:-1])
            for key in base_pars if key.endswith('}'))

    base_pars['initial_redshift'] = z_initial
    return add_pbh_population(base_pars, pop_id, pop_pars)


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------
def run_case(universe='clean', M_PBH_Msun=10.0, f_PBH=9.76e-3,
             lam=0.01, beta=0.2, f_E=0.15, sed_path=DEFAULT_SED_PATH,
             extra_ares_kwargs=None, return_sim=False):
    """
    Run one ARES thermal/ionization history in the chosen universe.
    """
    setup = UNIVERSES[universe]
    z_initial = setup['z_initial']

    pop_pars, model = make_pbh_population(
        M_PBH_Msun=M_PBH_Msun,
        f_PBH=f_PBH,
        lam=lam,
        beta=beta,
        f_E=f_E,
        sed_path=sed_path,
    )

    if universe == 'clean':
        pars = clean_pars(pop_pars, z_initial)
    else:
        pars = full_pars(pop_pars, z_initial)

    pars['verbose'] = False
    pars['include_He'] = setup['include_He']
    pars['solver_atol'] = 1e-12
    pars['solver_rtol'] = 1e-10

    pars.update(dict(
        initial_redshift=z_initial,
        final_redshift=Z_FINAL,
    ))

    if extra_ares_kwargs:
        pars.update(extra_ares_kwargs)

    sim = ares.simulations.Global21cm(**pars)
    sim.run()

    z = np.asarray(sim.history['z'])
    Tk = np.asarray(sim.history['igm_Tk'])
    dTb = np.asarray(sim.history['dTb'])

    if 'igm_e' in sim.history:
        xe = np.asarray(sim.history['igm_e'])
    else:
        xe = 1.0 - np.asarray(sim.history['igm_h_1'])
        print("No igm_e gave by ARES")

    result = z, Tk, xe, dTb, model

    if return_sim:
        result  += (sim,)

    return result

def Tk_at_z(z_arr, Tk_arr, z_target=17.0):
    """Interpolate the T_k(z) history onto a single target redshift."""
    # np.interp needs increasing x, so sort just in case
    order = np.argsort(z_arr)
    return float(np.interp(z_target, z_arr[order], Tk_arr[order]))

def figure3(universe='clean', M_PBH_Msun=10.0, lam=0.01,
            f_PBH=.01, f_E=0.15, betas=(0.2, 0.4, 0.6, 0.8, 1.0)):
    setup = UNIVERSES[universe]
    z_initial = setup['z_initial']
    prefix = setup['file_prefix']
    outpath = os.path.join(OUTPUT_DIR, f'{prefix}_fig3_reproduction.png')
    outpath_gs = os.path.join(OUTPUT_DIR, f'{prefix}_fig3_global_signal.png')

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig_gs, ax_gs = plt.subplots(figsize=(6.5, 5.5))
    # f_PBH is a free parameter in Hektor et al. It can be one specific value for each beta

    # LambdaCDM baseline (f_PBH = 0 -> no PBH heating/ionization at all)
    z0, Tk0, xe0, dTb0, _ = run_case(universe, M_PBH_Msun=M_PBH_Msun,
                                     f_PBH=0.0, lam=lam, beta=0.2, f_E=f_E)
    ax.plot(z0, Tk0, color='#4169E1', lw=2, label=r'$\Lambda$CDM')
    ax_gs.plot(z0, dTb0, color='k', lw=2, label=r'$\Lambda$CDM')

    Tk0_17 = Tk_at_z(z0, Tk0, 17.2)
    print(f"[{universe}] LambdaCDM Tk(z=17.2) = {Tk0_17:.2f} K  (paper quotes ~6.8-7 K)")

    # CMB temperature for reference, simply T_0(1+z)
    z_cmb = np.logspace(np.log10(Z_FINAL), np.log10(1000), 200)
    ax.plot(z_cmb, 2.725 * (1.0 + z_cmb), color='gray', ls='-', lw=1,
            label=r'$T_{\rm CMB}$')

    #cmap = plt.get_cmap('viridis')
    # Trying to get as close as possible similar colors
    beta_colors = {
        0.2: '#66C2A4',
        0.4: '#3AAE8C',
        0.6: '#39A852',
        0.8: '#4DBBD5',
        1.0: '#4169E1',
    }

    for i, beta in enumerate(betas):
        z, Tk, xe, dTb, model = run_case(
            universe,
            M_PBH_Msun=M_PBH_Msun,
            f_PBH=f_PBH,
            lam=lam,
            beta=beta,
            f_E=f_E,
        )

        Tk_172 = Tk_at_z(z, Tk, 17.2)

        print(
            f"[{universe}] beta = {beta:.1f}, "
            f"f_PBH = {f_PBH:.5e}, "
            f"Tk(z=17.2) = {Tk_172:.2f} K"
        )

        ax.plot(
            z, Tk,
            color=beta_colors[beta],
            lw=1.5,
            label=fr'$\beta = {beta:.1f}$'
        )

        ax_gs.plot(
            z, dTb,
            color=beta_colors[beta],
            lw=1.5,
            label=fr'$\beta={beta:.1f}$'
        )


    ax.axvspan(15, 21, color='gray', alpha=0.15, lw=0)
    ax.axhline(7.0, color='C3', ls=':', lw=1)
    ax.text(z_initial * 0.6, 7.3, '7 K ' + r'$\Lambda$CDM', fontsize=8, color='C3')
    ax.axhline(4.0, color='C0', ls=':', lw=1)
    ax.text(z_initial * 0.6, 4.3, '4 K EDGES', fontsize=8, color='C0')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 1000)
    ax.set_ylim(1, 2.725 * (1.0 + 1000))

    # The sort of graph Hektor et al presents
    ax.set_xlabel('z')
    ax.set_ylabel(r'$T_k$ [K]')
    ax.set_title(fr'Reproduction of Fig. 3 ({universe}): $M_{{\rm PBH}}={M_PBH_Msun:.0f}\,M_\odot$, 'fr'$\lambda={lam}$')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")

    # What we can actually measure with Global 21 cm line
    ax_gs.set_xlabel('z')
    ax_gs.set_ylabel(r'$\delta T_b$ [mK]')
    ax_gs.set_title(fr'Global 21-cm signal ({universe}): 'fr'$M_{{\rm PBH}}={M_PBH_Msun:.0f}\,M_\odot$, '  fr'$\lambda={lam}$')
    ax_gs.legend(fontsize=8, ncol=2)
    ax_gs.set_xlim(100, 5)
    fig_gs.tight_layout()
    fig_gs.savefig(outpath_gs, dpi=150)
    print(f"Saved {outpath_gs}")
    return outpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('universe', nargs='?', default='clean',
                        choices=['clean', 'full', 'both'])
    args = parser.parse_args()

    if not os.path.exists(DEFAULT_SED_PATH):
        print(f"{DEFAULT_SED_PATH} not found -- check the Hektor SED path.")
        sys.exit(1)

    universes = ['clean', 'full'] if args.universe == 'both' else [args.universe]
    for universe in universes:
        figure3(universe)
