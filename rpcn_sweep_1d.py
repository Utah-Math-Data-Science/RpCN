import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    import torch

    # ttg_1d.py and the helper modules live next to this notebook.
    if str(mo.notebook_dir()) not in sys.path:
        sys.path.insert(0, str(mo.notebook_dir()))
    from ttg_1d import build_problem, measure, rpcn_nfe, run_rpcn

    pt = 1/72.27
    page_width = 397.48499*pt
    golden_ratio = (1 + 5**0.5) / 2
    try:
        plt.style.use('style.mplstyle')
    except (RuntimeError, OSError):
        plt.rcParams.update({
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
        })


@app.function
def parse_int_list(text):
    return sorted({int(s.strip()) for s in text.split(',') if s.strip()})


@app.cell
def _():
    sweep_device = 'cuda'
    sweep_rng_seed = mo.ui.number(value=31, label='RNG seed')
    sweep_n_gmm_modes = mo.ui.number(start=1, step=1, value=1, label='Num. GMM modes')
    sweep_x_spacing = mo.ui.number(start=0.0, step=0.5, value=3.0, label='x_spacing')
    sweep_tilt = mo.ui.number(start=0, stop=25, step=1, value=1, label=r'Reward tilt $\beta$')
    sweep_n_training_samples = mo.ui.number(start=128, stop=32768, step=128, value=9984, label='sm_n_training_samples')
    sweep_n_sampling_steps = mo.ui.number(start=20, stop=1000, step=10, value=50, label='sm_n_sampling_steps')
    sweep_batch_size = mo.ui.number(start=128, stop=32768, step=128, value=9984, label='Batch size')

    sweep_replicas = mo.ui.text(value='1,2,3,4', label='Replicas $K$')
    sweep_pcn_steps = mo.ui.text(value='5,7,9', label='pCN steps $M$')
    sweep_run = mo.ui.run_button(label='Run sweep')

    mo.vstack([
        mo.md('### Problem'),
        sweep_rng_seed,
        mo.hstack([sweep_n_gmm_modes, sweep_x_spacing, sweep_tilt]),
        mo.hstack([sweep_n_training_samples, sweep_n_sampling_steps, sweep_batch_size]),
        mo.md('### RpCN sweep'),
        mo.hstack([sweep_replicas, sweep_pcn_steps]),
        sweep_run,
    ])
    return (
        sweep_batch_size,
        sweep_device,
        sweep_n_gmm_modes,
        sweep_n_sampling_steps,
        sweep_n_training_samples,
        sweep_pcn_steps,
        sweep_replicas,
        sweep_rng_seed,
        sweep_run,
        sweep_tilt,
        sweep_x_spacing,
    )


@app.cell
def _(
    sweep_batch_size,
    sweep_device,
    sweep_n_gmm_modes,
    sweep_n_training_samples,
    sweep_rng_seed,
    sweep_tilt,
    sweep_x_spacing,
):
    sweep_problem = build_problem(
        rng_seed=sweep_rng_seed.value,
        n_gmm_modes=sweep_n_gmm_modes.value,
        x_spacing=sweep_x_spacing.value,
        tilt=sweep_tilt.value,
        n_training_samples=sweep_n_training_samples.value,
        eval_batch_size=sweep_batch_size.value,
        device=sweep_device,
    )
    return (sweep_problem,)


@app.cell
def _(sweep_n_sampling_steps, sweep_pcn_steps, sweep_replicas):
    # Planned cost of each (K, M) pair; NFE per sample is K S (1 + M), so it is not held fixed across the sweep.
    _S = sweep_n_sampling_steps.value
    _df_plan = pl.DataFrame([
        dict(replicas=k, pcn_steps=m, expected_nfe=rpcn_nfe(_S, k, m))
        for k in parse_int_list(sweep_replicas.value) for m in parse_int_list(sweep_pcn_steps.value)
    ])
    mo.vstack([
        mo.md(f'Expected NFE per sample ($K S (1 + M)$, $S = {_S}$); total {_df_plan["expected_nfe"].sum()}:'),
        _df_plan.pivot(on='pcn_steps', index='replicas', values='expected_nfe'),
    ])
    return


@app.cell
def _(
    sweep_batch_size,
    sweep_n_sampling_steps,
    sweep_pcn_steps,
    sweep_problem,
    sweep_replicas,
    sweep_run,
):
    mo.stop(not sweep_run.value, mo.md('Press **Run sweep** to start.'))

    def _run_sweep():
        u_grid = sweep_problem.u_grid(sweep_n_sampling_steps.value)
        pairs = [(k, m) for k in parse_int_list(sweep_replicas.value) for m in parse_int_list(sweep_pcn_steps.value)]
        rows = []
        for k, m in mo.status.progress_bar(pairs, title='RpCN sweep', show_eta=True, show_rate=True):
            _, metrics = measure(
                sweep_problem, run_rpcn, sweep_batch_size.value, u_grid=u_grid, n_replicas=k, pcn_steps=m,
            )
            rows.append(dict(replicas=k, pcn_steps=m, **metrics))
            torch.cuda.empty_cache()
        return pl.DataFrame(rows)

    df_sweep = _run_sweep()
    df_sweep
    return (df_sweep,)


@app.cell
def _(df_sweep):
    from plots import save_all_subfigures
    from pathlib import Path
    SAVE_DIR = Path('~/Overleaf/RpCN/imgs/toy_example').expanduser()
    def _heatmaps():
        _df_sweep = df_sweep.filter(pl.col('pcn_steps').is_in([1, 3]).not_())
        replicas = _df_sweep['replicas'].unique().sort().to_list()
        pcn_steps = _df_sweep['pcn_steps'].unique().sort().to_list()
        scale = 1e-2  # both heatmaps are in units of 10^-2
        # Font sizes (pt) for the half-page-width figure; cell values are smallest so 3 decimals fit a cell.
        title_size, label_size, annot_size = 8, 8, 8
        kl_max = _df_sweep['kl'].max() / scale
        panels = {
            # (title, cmap, fmt, extra heatmap kwargs)
            # Color range anchored at 0; vmax as a fraction of the largest KL sets the saturation.
            'kl': (r'KL divergence', 'Purples', '.3f', dict(vmin=0, vmax=.50 * kl_max)),
            'mean_reward': (r'Mean reward', 'Blues', '.3f', {}),
        }
        # One facet per metric, so save_all_subfigures writes each heatmap to its own file.
        df_long = _df_sweep.unpivot(
            index=['replicas', 'pcn_steps'], on=list(panels), variable_name='metric', value_name='value',
        )
        def draw(data, **kwargs):
            _, cmap, fmt, heatmap_kwargs = panels[data['metric'].iloc[0]]
            grid = data.pivot(index='replicas', columns='pcn_steps', values='value').reindex(index=replicas, columns=pcn_steps)
            sns.heatmap(
                grid / scale, ax=plt.gca(), cmap=cmap, annot=True, fmt=fmt,
                annot_kws=dict(fontsize=annot_size),
                xticklabels=pcn_steps, yticklabels=replicas, linewidths=.0, linecolor='white',
                cbar=None,
                **heatmap_kwargs,
            )
        # Both panels side by side within half the page width.
        fig_width = page_width / 2
        height = fig_width / golden_ratio
        g = sns.FacetGrid(
            df_long, col='metric', col_order=list(panels),
            height=height, aspect=fig_width / 2 / height,
            sharex=False, sharey=False,  # independent axes, so each invert_yaxis below applies once
            despine=False,
        )
        g.map_dataframe(draw)
        g.set_axis_labels('pCN steps $M$', 'Replicas $K+1$', fontsize=label_size)
        for ax, (title, *_) in zip(g.axes.flat, panels.values()):
            ax.set_title(title, fontsize=title_size)
            ax.invert_yaxis()
            ax.tick_params(axis='both', length=2, labelsize=label_size)
            ax.tick_params(axis='y', rotation=0)
            if 'mean reward' in title.lower():
                ax.set_yticks([])
        g.tight_layout()
        # Fixed small gap between the panels (as a fraction of the axes width), set after tight_layout so
        # overflowing cell annotations don't widen it.
        g.figure.subplots_adjust(wspace=0.01)
        return g
    _g = _heatmaps()
    save_all_subfigures(_g, SAVE_DIR/'Ablation')
    _g.figure
    return


if __name__ == "__main__":
    app.run()
