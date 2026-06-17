#!/usr/bin/env python3
"""
Real-data comparison: BayesOptGenerator vs GridWalk vs LegacyGridSearch on NGC6278.

Three free parameters: ml (mass-to-light), c-dh (NFW concentration, log-space),
f-dh (NFW mass ratio M200/M_stars, log-space). Stars shape fixed to avoid
triaxiality validity conflicts at batch_size=1.

Usage:
    python run_comparison_real.py [options]

    --input-dir DIR    Path to NGC6278_input/   [default: auto-detect next to script]
    --output-dir DIR   Root for all output       [default: ./comparison_YYYYMMDD_HHMMSS]
    --ncpus N          CPUs per generator run    [default: 4]
    --nmodels N        Model budget per gen      [default: 60]
    --nE N             Orblib energy shells      [default: 5]
    --nI2 N            Orblib I2 shells          [default: 5]
    --nI3 N            Orblib I3 shells          [default: 5]
    --dithering N      Orbit dithering           [default: 1]
    --generators LIST  Comma-separated subset    [default: bayesopt,gridwalk,legacygrid]
    --skip-runs        Skip model runs; re-plot existing results only
"""
import argparse
import datetime
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')  # headless — must be set before any pyplot import

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    here = pathlib.Path(__file__).parent.resolve()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', default=str(here / 'NGC6278_input'))
    p.add_argument('--output-dir', default=None)
    p.add_argument('--ncpus', type=int, default=4)
    p.add_argument('--nmodels', type=int, default=60)
    p.add_argument('--nE', type=int, default=5)
    p.add_argument('--nI2', type=int, default=5)
    p.add_argument('--nI3', type=int, default=5)
    p.add_argument('--dithering', type=int, default=1)
    p.add_argument('--generators', default='bayesopt,gridwalk,legacygrid')
    p.add_argument('--skip-runs', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def _system_block():
    """NGC6278 galaxy components. Free params: ml, c-dh, f-dh."""
    return {
        'system_attributes': {'distMPc': 39.96, 'name': 'NGC6278'},
        'system_components': {
            'bh': {
                'type': 'Plummer',
                'contributes_to_potential': True,
                'include': True,
                'parameters': {
                    'm': {
                        'fixed': True, 'value': 5.0, 'logarithmic': True,
                        'par_generator_settings': {
                            'lo': 1.0, 'hi': 10.0, 'step': 1.0, 'minstep': 0.0},
                        'LaTeX': '$\\log(M_\\mathrm{BH}/M_\\odot)$',
                    },
                    'a': {'fixed': True, 'value': 1.0e-3,
                          'LaTeX': '$a_\\mathrm{BH}$'},
                },
            },
            'dh': {
                'type': 'NFW',
                'contributes_to_potential': True,
                'include': True,
                'parameters': {
                    'c': {
                        'par_generator_settings': {
                            'lo': 2.0, 'hi': 4.0, 'step': 0.5, 'minstep': 0.1},
                        'logarithmic': True,
                        'fixed': False,   # FREE — log10(NFW concentration)
                        'value': 3.0,
                        'LaTeX': '$\\log(c_\\mathrm{NFW})$',
                    },
                    'f': {
                        'par_generator_settings': {
                            'lo': 0.0, 'hi': 2.0, 'step': 0.5, 'minstep': 0.1},
                        'logarithmic': True,
                        'fixed': False,   # FREE — log10(M200/M_stars)
                        'value': 1.0,
                        'LaTeX': '$\\log(M_{200}/M_\\star)$',
                    },
                },
            },
            'stars': {
                'type': 'TriaxialVisibleComponent',
                'contributes_to_potential': False,
                'mge_pot': 'mge.ecsv',
                'mge_lum': 'mge.ecsv',
                'include': True,
                'parameters': {
                    'q': {
                        'par_generator_settings': {
                            'lo': 0.05, 'hi': 0.99, 'step': 0.1, 'minstep': 0.02},
                        'fixed': True,  # fixed — avoids triaxiality validity conflicts
                        'value': 0.54,
                        'LaTeX': '$q_\\star$',
                    },
                    'p': {
                        'par_generator_settings': {
                            'lo': 0.99, 'hi': 0.999, 'step': 0.02, 'minstep': 0.01},
                        'fixed': True, 'value': 0.99,
                        'LaTeX': '$p_\\star$',
                    },
                    'u': {
                        'par_generator_settings': {
                            'lo': 0.95, 'hi': 1.0, 'step': 0.01, 'minstep': 0.01},
                        'fixed': True, 'value': 0.9999,
                        'LaTeX': '$u_\\star$',
                    },
                },
                'kinematics': {
                    'kinset1': {
                        'type': 'GaussHermite',
                        'hist_width': '2719.8215332031',
                        'hist_center': '0.0000',
                        'hist_bins': '203',
                        'datafile': 'gauss_hermite_kins.ecsv',
                        'aperturefile': 'aperture.dat',
                        'binfile': 'bins.dat',
                        'with_pops': False,
                    },
                },
            },
        },
        'system_parameters': {
            'ml': {
                'par_generator_settings': {
                    'lo': 1.0, 'hi': 9.0, 'step': 1.0, 'minstep': 0.5},
                'fixed': False,   # FREE
                'value': 5.0,
                'LaTeX': '$\\Upsilon_r$',
            },
        },
    }


def _shared_block(input_dir, output_dir, ncpus, nE, nI2, nI3, dithering):
    return {
        'orblib_settings': {
            'nE': nE, 'nI2': nI2, 'nI3': nI3,
            'logrmin': -0.101275, 'logrmax': 1.99123,
            'dithering': dithering,
            'quad_nr': 10, 'quad_nth': 6, 'quad_nph': 6,
            'orbital_periods': 200, 'sampling': 50000,
            'starting_orbit': 1, 'number_orbits': -1,
            'accuracy': '1.0d-5', 'random_seed': 42,
        },
        'weight_solver_settings': {
            'type': 'NNLS', 'nnls_solver': 'scipy',
            'regularisation': 0, 'number_GH': 4,
            'GH_sys_err': '0.0 0.0 0.0 0.0 0.3 0.3 0.6 0.6',
            'lum_intr_rel_err': 0.01, 'sb_proj_rel_err': 0.02,
            'reattempt_failures': True,
        },
        'legacy_settings': {'directory': 'default'},
        'io_settings': {
            'input_directory': str(pathlib.Path(input_dir).resolve()) + '/',
            'output_directory': str(pathlib.Path(output_dir).resolve()) + '/',
            'all_models_file': 'all_models.ecsv',
        },
        'multiprocessing_settings': {
            'ncpus': ncpus,
            'modeliterator': 'ModelInnerIterator',
            'use_jax_orblib': False,
        },
    }


def _parspace_bayesopt(nmodels, ncpus):
    # n_initial_random=12 gives 4 random draws per free parameter before the GP fits.
    # discretize_non_ml_params snaps c-dh (step=0.5) and f-dh (step=0.5) to grid
    # points after acquisition, so revisiting the same orbital config reuses the orblib.
    # min_delta_chi2_abs=-1e6 disables the chi2-improvement stopping criterion so
    # BayesOpt runs its full n_max_mods budget (it would fire prematurely after
    # early Sobol draws where the best chi2 barely moves between iterations).
    return {
        'parameter_space_settings': {
            'generator_type': 'BayesOptGenerator',
            'which_chi2': 'kinchi2',
            'generator_settings': {
                'warmup_mode': 'sobol',
                'n_initial_random': 12,
                'batch_size': ncpus,
                'n_orblib_configs': ncpus,
                'n_ml_per_config': 1,
                'discretize_non_ml_params': True,
            },
            'stopping_criteria': {
                'min_delta_chi2_abs': -1e6,
                'n_max_mods': nmodels,
                'n_max_iter': 200,
            },
        },
    }


def _parspace_gridwalk(nmodels):
    return {
        'parameter_space_settings': {
            'generator_type': 'GridWalk',
            'which_chi2': 'kinchi2',
            'generator_settings': {
                'step': 1.0,
                'minstep': 0.1,
            },
            'stopping_criteria': {
                'min_delta_chi2_abs': 0.5,
                'n_max_mods': nmodels,
                'n_max_iter': 200,
            },
        },
    }


def _parspace_legacygrid(nmodels):
    return {
        'parameter_space_settings': {
            'generator_type': 'LegacyGridSearch',
            'which_chi2': 'kinchi2',
            'generator_settings': {
                'threshold_del_chi2': 4.0,
            },
            'stopping_criteria': {
                'min_delta_chi2_abs': 0.5,
                'n_max_mods': nmodels,
                'n_max_iter': 200,
            },
        },
    }


def build_config(gen_name, input_dir, output_dir, ncpus, nmodels,
                 nE, nI2, nI3, dithering):
    cfg = {}
    cfg.update(_system_block())
    cfg.update(_shared_block(input_dir, output_dir, ncpus, nE, nI2, nI3, dithering))
    if gen_name == 'bayesopt':
        cfg.update(_parspace_bayesopt(nmodels, ncpus))
    elif gen_name == 'gridwalk':
        cfg.update(_parspace_gridwalk(nmodels))
    elif gen_name == 'legacygrid':
        cfg.update(_parspace_legacygrid(nmodels))
    else:
        raise ValueError(f'Unknown generator: {gen_name!r}')
    return cfg


# ---------------------------------------------------------------------------
# Run one generator
# ---------------------------------------------------------------------------

def run_generator(gen_name, cfg, gen_outdir, cfg_path):
    from dynamite import model_iterator, config_reader

    all_models_path = pathlib.Path(gen_outdir) / 'all_models.ecsv'
    if all_models_path.exists():
        print(f'\n[{gen_name}] already complete at {gen_outdir} — skipping.')
        return

    pathlib.Path(gen_outdir).mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'w') as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    print(f'\n{"=" * 60}')
    print(f'[{gen_name}] config → {cfg_path}')
    print(f'[{gen_name}] output → {gen_outdir}')
    print(f'{"=" * 60}', flush=True)

    c = config_reader.Configuration(str(cfg_path), reset_logging=True)
    model_iterator.ModelIterator(c, plots=False)

    done = int(np.sum(np.asarray(c.all_models.table['all_done'], dtype=bool)))
    print(f'[{gen_name}] finished — {done} models completed.', flush=True)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

FREE_PARAMS  = ['ml', 'c-dh', 'f-dh']
LOG_PARAMS   = {'c-dh', 'f-dh'}   # stored as linear in all_models; take log10 before plotting
PARAM_LATEX  = {
    'ml':    r'$\Upsilon_r$',
    'c-dh':  r'$\log(c_\mathrm{NFW})$',
    'f-dh':  r'$\log(M_{200}/M_\star)$',
}
GEN_LABELS = {
    'bayesopt':   'BayesOpt (GP + qLogEI)',
    'gridwalk':   'GridWalk',
    'legacygrid': 'LegacyGridSearch',
}
GEN_COLORS = {
    'bayesopt':   '#1f77b4',
    'gridwalk':   '#ff7f0e',
    'legacygrid': '#2ca02c',
}


def load_table(gen_outdir):
    from astropy.table import Table
    path = pathlib.Path(gen_outdir) / 'all_models.ecsv'
    if not path.exists():
        return None
    t = Table.read(str(path))
    done   = np.asarray(t['all_done'], dtype=bool)
    finite = np.isfinite(np.asarray(t['kinchi2'], dtype=float))
    return t[done & finite]


def _colvals(t, param):
    """Extract physical-space values for a free parameter column."""
    if param in t.colnames:
        return np.asarray(t[param], dtype=float)
    # fallback: strip component suffix
    base = param.split('-')[0]
    return np.asarray(t[base], dtype=float)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_corner_plot(tables, outpath):
    import matplotlib.pyplot as plt

    pairs = [('ml', 'c-dh'), ('ml', 'f-dh'), ('c-dh', 'f-dh')]
    gens  = [g for g in ('bayesopt', 'gridwalk', 'legacygrid') if tables.get(g) is not None]
    n_gen, n_pair = len(gens), len(pairs)

    fig, axes = plt.subplots(n_gen, n_pair,
                             figsize=(4.5 * n_pair, 3.8 * n_gen),
                             squeeze=False)
    fig.suptitle(
        'NGC6278  —  Proposed models by generator\n'
        'colour = model index (blue → early, red → late)',
        fontsize=12, y=1.02)

    for row, gen in enumerate(gens):
        t = tables[gen]
        n = len(t)
        for col, (xp, yp) in enumerate(pairs):
            ax = axes[row][col]
            xv = _colvals(t, xp)
            yv = _colvals(t, yp)
            if xp in LOG_PARAMS:
                xv = np.log10(np.asarray(xv, dtype=float))
            if yp in LOG_PARAMS:
                yv = np.log10(np.asarray(yv, dtype=float))
            sc = ax.scatter(xv, yv, c=np.arange(n), cmap='coolwarm',
                            s=45, alpha=0.85, edgecolors='k', linewidths=0.3,
                            vmin=0, vmax=max(n - 1, 1), zorder=3)
            best_i = int(np.argmin(np.asarray(t['kinchi2'], dtype=float)))
            ax.scatter([xv[best_i]], [yv[best_i]], marker='*', s=280,
                       color='gold', edgecolors='k', linewidths=0.8, zorder=5)
            if col == 0:
                ax.set_ylabel(GEN_LABELS.get(gen, gen), fontsize=9, labelpad=4)
            if row == n_gen - 1:
                ax.set_xlabel(PARAM_LATEX.get(xp, xp), fontsize=10)
            if row == 0:
                ax.set_title(
                    f'{PARAM_LATEX.get(xp, xp)} vs {PARAM_LATEX.get(yp, yp)}',
                    fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.4)
        cb = fig.colorbar(sc, ax=axes[row], shrink=0.75, pad=0.02)
        cb.set_label('Model index', fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {outpath}')


def make_convergence_plot(tables, outpath):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gen in ('bayesopt', 'gridwalk', 'legacygrid'):
        t = tables.get(gen)
        if t is None or len(t) == 0:
            continue
        chi2 = np.asarray(t['kinchi2'], dtype=float)
        ax.plot(np.arange(1, len(chi2) + 1),
                np.minimum.accumulate(chi2),
                label=GEN_LABELS.get(gen, gen),
                color=GEN_COLORS.get(gen),
                lw=2.0, marker='o', markersize=3.5, alpha=0.85)

    ax.set_xlabel('Cumulative models evaluated', fontsize=11)
    ax.set_ylabel(r'Running minimum $\chi^2_\mathrm{kin}$', fontsize=11)
    ax.set_title(r'NGC6278 — Convergence (3 free params: $\Upsilon_r$, $\log c_\mathrm{NFW}$, $\log(M_{200}/M_\star)$)',
                 fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {outpath}')


def make_chi2_surfaces_plot(tables, outpath):
    """Marginal chi2 vs each free parameter, coloured by model index."""
    import matplotlib.pyplot as plt

    gens = [g for g in ('bayesopt', 'gridwalk', 'legacygrid')
            if tables.get(g) is not None]
    n_gen, n_par = len(gens), len(FREE_PARAMS)

    fig, axes = plt.subplots(n_gen, n_par,
                             figsize=(4.5 * n_par, 3.5 * n_gen),
                             squeeze=False)
    fig.suptitle('NGC6278  —  Marginal $\\chi^2$ surfaces',
                 fontsize=12, y=1.02)

    for row, gen in enumerate(gens):
        t = tables[gen]
        n = len(t)
        chi2 = np.asarray(t['kinchi2'], dtype=float)
        best_i = int(np.argmin(chi2))

        for col, par in enumerate(FREE_PARAMS):
            ax = axes[row][col]
            xv = _colvals(t, par)
            if par in LOG_PARAMS:
                xv = np.log10(np.asarray(xv, dtype=float))
            ax.scatter(xv, chi2, c=np.arange(n), cmap='coolwarm',
                       s=35, alpha=0.75, edgecolors='k', linewidths=0.3,
                       vmin=0, vmax=max(n - 1, 1))
            ax.axvline(xv[best_i], color='gold', lw=1.5, ls='--', alpha=0.9)
            if col == 0:
                ax.set_ylabel(
                    GEN_LABELS.get(gen, gen) + r' — $\chi^2_\mathrm{kin}$',
                    fontsize=8)
            if row == n_gen - 1:
                ax.set_xlabel(PARAM_LATEX.get(par, par), fontsize=10)
            if row == 0:
                ax.set_title(PARAM_LATEX.get(par, par), fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {outpath}')


def print_summary(tables):
    print(f'\n{"=" * 65}')
    print('SUMMARY — NGC6278 real orblib comparison (ml, c-dh, f-dh free)')
    print(f'{"=" * 65}')
    hdr = f'{"Generator":<26} {"N models":>9} {"Best kinchi2":>13} {"ml":>6} {"log c":>7} {"log f":>7}'
    print(hdr)
    print('-' * 65)
    for gen in ('bayesopt', 'gridwalk', 'legacygrid'):
        t = tables.get(gen)
        if t is None or len(t) == 0:
            print(f'{GEN_LABELS.get(gen, gen):<26}  (no results)')
            continue
        chi2 = np.asarray(t['kinchi2'], dtype=float)
        i = int(np.argmin(chi2))
        ml_best = float(_colvals(t, 'ml')[i])
        c_best  = np.log10(float(_colvals(t, 'c-dh')[i]))  # stored as linear; table in log10
        f_best  = np.log10(float(_colvals(t, 'f-dh')[i]))
        print(f'{GEN_LABELS.get(gen, gen):<26} {len(t):>9} '
              f'{chi2[i]:>13.2f} {ml_best:>6.2f} {c_best:>7.3f} {f_best:>7.3f}')
    print(f'{"=" * 65}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        here  = pathlib.Path(__file__).parent.resolve()
        args.output_dir = str(here / f'comparison_{stamp}')
    outroot   = pathlib.Path(args.output_dir)
    input_dir = pathlib.Path(args.input_dir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        sys.exit(f'ERROR: input directory not found: {input_dir}')

    # Ensure dynamite is importable
    dyn_root = pathlib.Path(__file__).parent.parent
    if str(dyn_root) not in sys.path:
        sys.path.insert(0, str(dyn_root))

    generators = [g.strip() for g in args.generators.split(',')]
    print(f'Output:     {outroot}')
    print(f'Generators: {generators}')
    print(f'ncpus={args.ncpus}  nmodels={args.nmodels}  '
          f'orblib=({args.nE},{args.nI2},{args.nI3}) dithering={args.dithering}')

    # --- Run generators sequentially ---
    gen_outdirs = {}
    for gen in generators:
        gen_outdir = outroot / gen
        gen_outdirs[gen] = gen_outdir
        cfg_path = outroot / f'config_{gen}.yaml'
        cfg = build_config(gen, str(input_dir), str(gen_outdir),
                           args.ncpus, args.nmodels,
                           args.nE, args.nI2, args.nI3, args.dithering)
        if not args.skip_runs:
            run_generator(gen, cfg, str(gen_outdir), str(cfg_path))
        else:
            print(f'[{gen}] --skip-runs active.')

    # --- Load & summarise ---
    tables = {gen: load_table(gen_outdirs[gen]) for gen in generators}
    print_summary(tables)

    # --- Plots ---
    nonempty = {g: t for g, t in tables.items() if t is not None and len(t) > 0}
    if nonempty:
        make_corner_plot(nonempty, outroot / 'corner_proposals.png')
        make_convergence_plot(nonempty, outroot / 'convergence.png')
        make_chi2_surfaces_plot(nonempty, outroot / 'chi2_surfaces.png')
    else:
        print('No completed models found — skipping plots.')

    print(f'\nAll done.  Results in: {outroot}')


if __name__ == '__main__':
    main()
