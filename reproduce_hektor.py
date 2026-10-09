"""
Reproduce Hektor/ARES results.

Usage:
    python reproduce_hektor.py fig3
"""

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


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------
Z_INITIAL = 60 # ARES can handle up to 400 without problems
Z_FINAL = 6.0

def run_case(M_PBH_Msun=10.0, f_PBH=9.76e-3, lam=0.01, beta=0.2, f_E=0.15, sed_path=DEFAULT_SED_PATH, extra_ares_kwargs=None, return_sim=False):
    """
    Run one ARES thermal/ionization history with ONLY the Hektor PBH source.
    """

    pop_pars, model = make_pbh_population(
        M_PBH_Msun=M_PBH_Msun,
        f_PBH=f_PBH,
        lam=lam,
        beta=beta,
        f_E=f_E,
        sed_path=sed_path,
    )

    pars = dict(DEFAULT_COSM_KWARGS)
    pars['problem_type'] = 100 # Clean universe
    pars['verbose'] = False
    pars.update(pop_pars)
    pars['pop_zform'] = 1000 # When the PBH appear, not sure exactly what it affects yet, currently testing
    pars['first_light_redshift'] = Z_INITIAL # When the PBH start evolving
    pars['solver_atol'] = 1e-12
    pars['solver_rtol'] = 1e-10
    pars.update(PBH_SOURCE_FLAGS)
    pars.update(dict(
        initial_redshift=Z_INITIAL,
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

def figure3(M_PBH_Msun=10.0, lam=0.01, f_PBH=.01, f_E=0.15,
            betas=(0.2, 0.4, 0.6, 0.8, 1.0),
            outpath= os.path.join(OUTPUT_DIR, 'hektor_fig3_reproduction.png')):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig_gs, ax_gs = plt.subplots(figsize=(6.5, 5.5))
    # f_PBH is a free parameter in Hektor et al. It can be one specific value for each beta

    # LambdaCDM baseline (f_PBH = 0 -> no PBH heating/ionization at all)
    z0, Tk0, xe0, dTb0, _ = run_case(M_PBH_Msun=M_PBH_Msun, f_PBH=0.0, lam=lam, beta=0.2, f_E=f_E)
    ax.plot(z0, Tk0, color='#4169E1', lw=2, label=r'$\Lambda$CDM')
    ax_gs.plot(z0, dTb0, color='k', lw=2, label=r'$\Lambda$CDM')

    Tk0_17 = Tk_at_z(z0, Tk0, 17.2)
    print(f"LambdaCDM Tk(z=17.2) = {Tk0_17:.2f} K  (paper quotes ~6.8-7 K)")

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
            M_PBH_Msun=M_PBH_Msun,
            f_PBH=f_PBH,
            lam=lam,
            beta=beta,
            f_E=f_E,
        )

        Tk_172 = Tk_at_z(z, Tk, 17.2)

        print(
            f"beta = {beta:.1f}, "
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
    ax.text(Z_INITIAL * 0.6, 7.3, '7 K ' + r'$\Lambda$CDM', fontsize=8, color='C3')
    ax.axhline(4.0, color='C0', ls=':', lw=1)
    ax.text(Z_INITIAL * 0.6, 4.3, '4 K EDGES', fontsize=8, color='C0')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 1000)
    ax.set_ylim(1, 2.725 * (1.0 + 1000))

    # The sort of graph Hektor et al presents
    ax.set_xlabel('z')
    ax.set_ylabel(r'$T_k$ [K]')
    ax.set_title(fr'Reproduction of Fig. 3: $M_{{\rm PBH}}={M_PBH_Msun:.0f}\,M_\odot$, 'fr'$\lambda={lam}$')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")

    # What we can actually measure with Global 21 cm line
    ax_gs.set_xlabel('z')
    ax_gs.set_ylabel(r'$\delta T_b$ [mK]')
    ax_gs.set_title(fr'Global 21-cm signal: 'fr'$M_{{\rm PBH}}={M_PBH_Msun:.0f}\,M_\odot$, '  fr'$\lambda={lam}$')
    ax_gs.legend(fontsize=8, ncol=2)
    ax_gs.set_xlim(100, 5)
    fig_gs.tight_layout()

    outpath_gs = os.path.join(
        OUTPUT_DIR,
        'hektor_fig3_global_signal.png'
    )
    fig_gs.savefig(outpath_gs, dpi=150)
    print(f"Saved {outpath_gs}")
    return outpath

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if not os.path.exists(DEFAULT_SED_PATH):
        print(f"{DEFAULT_SED_PATH} not found -- check the Hektor SED path.")
        sys.exit(1)

    if mode in ('fig3', 'both'):
        figure3()
