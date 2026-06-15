"""Corner-style plot comparing how BayesOpt, GridWalk, and LegacyGridSearch
explore a 2D parameter space (q-stars × ml).

Synthetic chi2 landscape:
    chi2(q, ml) = 80*(q - 0.4)^2 + 200*(ml - 5.5)^2 + 15
    True minimum: q=0.4, ml=5.5, chi2=15

The plot shows proposed models as scatter points (one panel per generator pair),
colored by iteration number, so that BayesOpt's exploitation can be compared
with the grid methods' uniform coverage.

Run from dev_tests/:
    cd /Users/pesmith/research/dynamite/dev_tests
    /opt/miniconda3/envs/main/bin/python3 plot_generator_comparison.py
"""
import os
import sys
import copy
import shutil
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '..')
import dynamite as dyn

# ---------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------
Q_TRUE  = 0.4
ML_TRUE = 5.5
CHI2_MIN = 15.0

Q_LO, Q_HI   = 0.05, 0.99
ML_LO, ML_HI = 1.0, 9.0


def dummy_chi2_2d(parset):
    q  = float(parset['q-stars'])
    ml = float(parset['ml'])
    return 80.0 * (q - Q_TRUE)**2 + 200.0 * (ml - ML_TRUE)**2 + CHI2_MIN


# ---------------------------------------------------------------------------
# Run a generator via its YAML config in dummy mode, return (table, cfg)
# ---------------------------------------------------------------------------
def run_generator(config_file, label, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    c = dyn.config_reader.Configuration(config_file)
    dyn.model_iterator.ModelIterator(
        c,
        do_dummy_run=True,
        dummy_chi2_function=dummy_chi2_2d,
        plots=False,
    )
    t = c.all_models.table
    print(f'  {label}: {len(t)} models, '
          f'best chi2={float(np.nanmin(t["kinchi2"])):.2f}')
    return t


# ---------------------------------------------------------------------------
# Build temporary YAML configs for 2-D (q, ml) by modifying the base BayesOpt
# config. All generators share the same stopping criteria so comparisons are fair.
# ---------------------------------------------------------------------------
import yaml

# Stopping: run until n_max_mods so the full trajectory is visible in plots.
# min_delta_chi2_abs is set very low to prevent early exit before the model
# budget is exhausted.
STOPPING = {'min_delta_chi2_abs': -1e9, 'n_max_mods': 24, 'n_max_iter': 20}


def make_2d_config(generator_type, extra_settings, out_dir, base_yaml='bayesopt_qml_modelinner.yaml'):
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg['parameter_space_settings']['generator_type'] = generator_type
    cfg['parameter_space_settings'].pop('generator_settings', None)
    cfg['parameter_space_settings']['stopping_criteria'] = STOPPING
    cfg['parameter_space_settings'].update(extra_settings)
    cfg['io_settings']['output_directory'] = out_dir
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                      delete=False, dir='.')
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Run all generators
# ---------------------------------------------------------------------------
print('Running BayesOpt (q+ml, 2D) ...')
bo_cfg = make_2d_config(
    'BayesOptGenerator',
    {'generator_settings': {'n_initial_random': 8, 'batch_size': 4,
                            'n_orblib_configs': 4, 'n_ml_per_config': 1}},
    'NGC6278_bayesopt_qml_output/',
)
bo_table = run_generator(bo_cfg, 'BayesOpt', 'NGC6278_bayesopt_qml_output')
os.unlink(bo_cfg)

print('Running GridWalk (q+ml, 2D) ...')
gw_cfg = make_2d_config('GridWalk', {}, 'NGC6278_gridwalk_qml_output/')
gw_table = run_generator(gw_cfg, 'GridWalk', 'NGC6278_gridwalk_qml_output')
os.unlink(gw_cfg)

print('Running LegacyGridSearch (q+ml, 2D) ...')
lg_cfg = make_2d_config('LegacyGridSearch',
                         {'generator_settings': {'threshold_del_chi2': 0.1}},
                         'NGC6278_legacygrid_qml_output/')
lg_table = run_generator(lg_cfg, 'LegacyGridSearch', 'NGC6278_legacygrid_qml_output')
os.unlink(lg_cfg)


# ---------------------------------------------------------------------------
# Extract coordinates, chi2, iteration for each generator
# ---------------------------------------------------------------------------
def extract(t):
    q   = np.asarray(t['q-stars'], dtype=float)
    ml  = np.asarray(t['ml'], dtype=float)
    chi2 = np.asarray(t['kinchi2'], dtype=float)
    itr  = np.asarray(t['which_iter'], dtype=int)
    return q, ml, chi2, itr


bo_q, bo_ml, bo_chi2, bo_iter = extract(bo_table)
gw_q, gw_ml, gw_chi2, gw_iter = extract(gw_table)
lg_q, lg_ml, lg_chi2, lg_iter = extract(lg_table)

generators = [
    ('BayesOpt',         bo_q, bo_ml, bo_chi2, bo_iter),
    ('GridWalk',         gw_q, gw_ml, gw_chi2, gw_iter),
    ('LegacyGridSearch', lg_q, lg_ml, lg_chi2, lg_iter),
]

all_iters = max(max(bo_iter), max(gw_iter), max(lg_iter))
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
MARKERS = ['o', 's', '^']

# ---------------------------------------------------------------------------
# Build iteration colormap (shared across generators)
# ---------------------------------------------------------------------------
iter_cmap = cm.viridis
iter_norm = mcolors.Normalize(vmin=0, vmax=all_iters)


# ---------------------------------------------------------------------------
# Figure layout:
#   Top row (3 panels): q vs ml scatter for each generator, colored by iter
#   Bottom-left: q marginals (1D histograms) overlaid
#   Bottom-right: ml marginals (1D histograms) overlaid
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Parameter-space exploration: BayesOpt vs grid methods\n'
             r'($q_{\rm stars}$ × $m_l$, dummy $\chi^2$ landscape)',
             fontsize=13)

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                      left=0.07, right=0.93, top=0.90, bottom=0.08)

# --- 2D scatter panels (top row) ---
scatter_axes = []
for col, (label, q, ml, chi2, itr) in enumerate(generators):
    ax = fig.add_subplot(gs[0, col])
    scatter_axes.append(ax)

    # Background landscape contour
    qg  = np.linspace(Q_LO,  Q_HI,  200)
    mlg = np.linspace(ML_LO, ML_HI, 200)
    QQ, MM = np.meshgrid(qg, mlg)
    ZZ = 80*(QQ - Q_TRUE)**2 + 200*(MM - ML_TRUE)**2 + CHI2_MIN
    ax.contourf(QQ, MM, ZZ, levels=np.logspace(np.log10(CHI2_MIN),
                np.log10(float(np.nanmax(chi2))+10), 12),
                cmap='Greys', alpha=0.35)
    ax.contour(QQ, MM, ZZ,
               levels=[CHI2_MIN, CHI2_MIN+5, CHI2_MIN+20, CHI2_MIN+100],
               colors='grey', linewidths=0.5, alpha=0.6)

    # True minimum
    ax.plot(Q_TRUE, ML_TRUE, 'r*', ms=12, zorder=5, label='true min')

    # Proposed models colored by iteration
    sc = ax.scatter(q, ml, c=itr, cmap=iter_cmap, norm=iter_norm,
                    s=50, edgecolors='k', linewidths=0.4, zorder=4,
                    label='proposed')

    ax.set_xlim(Q_LO, Q_HI)
    ax.set_ylim(ML_LO, ML_HI)
    ax.set_xlabel(r'$q_{\rm stars}$', fontsize=10)
    ax.set_ylabel(r'$m_l$', fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.text(0.97, 0.97, f'N={len(q)}\nbest χ²={float(np.nanmin(chi2)):.1f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Colorbar for iteration
sm = cm.ScalarMappable(cmap=iter_cmap, norm=iter_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=scatter_axes, orientation='vertical',
                    fraction=0.02, pad=0.02)
cbar.set_label('Iteration', fontsize=10)

# --- q marginals (bottom-left) ---
ax_q = fig.add_subplot(gs[1, 0:2])
for (label, q, ml, chi2, itr), col in zip(generators, COLORS):
    ax_q.scatter(q, itr + np.random.default_rng(0).uniform(-0.15, 0.15, len(q)),
                 c=col, alpha=0.7, s=30, marker=MARKERS[generators.index((label, q, ml, chi2, itr))],
                 label=label, edgecolors='k', linewidths=0.3)
ax_q.axvline(Q_TRUE, color='red', ls='--', lw=1.5, label='true min')
ax_q.set_xlabel(r'$q_{\rm stars}$', fontsize=10)
ax_q.set_ylabel('Iteration (jittered)', fontsize=10)
ax_q.set_title(r'$q_{\rm stars}$ proposals vs iteration', fontsize=10)
ax_q.set_xlim(Q_LO, Q_HI)
ax_q.legend(fontsize=8, loc='upper right')

# --- ml marginals (bottom-right) ---
ax_ml = fig.add_subplot(gs[1, 2])
for (label, q, ml, chi2, itr), col in zip(generators, COLORS):
    ax_ml.scatter(ml, itr + np.random.default_rng(1).uniform(-0.15, 0.15, len(ml)),
                  c=col, alpha=0.7, s=30, marker=MARKERS[generators.index((label, q, ml, chi2, itr))],
                  label=label, edgecolors='k', linewidths=0.3)
ax_ml.axvline(ML_TRUE, color='red', ls='--', lw=1.5, label='true min')
ax_ml.set_xlabel(r'$m_l$', fontsize=10)
ax_ml.set_ylabel('Iteration (jittered)', fontsize=10)
ax_ml.set_title(r'$m_l$ proposals vs iteration', fontsize=10)
ax_ml.set_xlim(ML_LO, ML_HI)
ax_ml.legend(fontsize=8, loc='upper right')

out_path = 'generator_corner_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved {out_path}')
plt.close(fig)

# ---------------------------------------------------------------------------
# Second figure: running-best chi2 per iteration (convergence curve)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
for (label, q, ml, chi2, itr), col, mk in zip(generators, COLORS, MARKERS):
    iters_sorted = sorted(set(itr))
    running_best = []
    cur = np.inf
    for it in iters_sorted:
        mask = itr == it
        finite = chi2[mask][np.isfinite(chi2[mask])]
        if len(finite):
            cur = min(cur, float(np.min(finite)))
        running_best.append(cur)
    ax2.plot(iters_sorted, running_best, marker=mk, color=col,
             label=label, ms=7, lw=2)

ax2.axhline(CHI2_MIN, color='red', ls='--', lw=1.5, label=f'true min ({CHI2_MIN})')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel(r'Running-best $\chi^2$', fontsize=11)
ax2.set_title('Convergence: running-best χ² per iteration', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

conv_path = 'generator_convergence.png'
fig2.savefig(conv_path, dpi=150, bbox_inches='tight')
print(f'Saved {conv_path}')
plt.close(fig2)

print('\nDone.')
