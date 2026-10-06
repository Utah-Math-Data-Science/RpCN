import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import duckdb
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from omegaconf import OmegaConf
    import hydra

    import conf.conf
    import userfm.plots


    pt = 1/72.27
    page_width = 397.48499*pt
    golden_ratio = (1 + 5**0.5) / 2
    try:
        plt.style.use('style.mplstyle')
    except RuntimeError:
        plt.rcParams.update({
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
        })


@app.cell
def _():
    SAVE_DIR = Path('~/Overleaf/RpCN/imgs/dynamical_systems/ConstraintHistogram').expanduser()
    SAVE_DIR
    return (SAVE_DIR,)


@app.cell
def _():
    duckdb_conn = duckdb.connect()
    duckdb_conn.execute(
        "ATTACH 'runs.sqlite' (TYPE SQLITE); use runs;",
    )
    return (duckdb_conn,)


@app.cell
def _():
    # dataset_names = ['Lorenz63', 'FitzHughNagumo']
    # dataset_name = dataset_names[0]
    dataset_constraint_values = pl.concat([
        pl.read_csv('Lorenz63SolverConstraintValues.csv').with_columns(dataset_name=pl.lit('Lorenz63')),
        pl.read_csv('FitzHughNagumoSolverConstraintValues.csv').with_columns(dataset_name=pl.lit('FitzHughNagumo')),
    ])
    dataset_constraint_values
    return (dataset_constraint_values,)


@app.cell
def _(dataset_constraint_values):
    dataset_histogram_bins = {}
    for _dataset_name in dataset_constraint_values['dataset_name'].unique():
        _ds = dataset_constraint_values.filter(dataset_name=_dataset_name)
        print(_ds['constraint_value'].max())
        dataset_histogram_bins[_dataset_name] = np.histogram(np.zeros(2), bins=128, range=(_ds['constraint_value'].min(), _ds['constraint_value'].max()))[1]
    dataset_histogram_bins
    return (dataset_histogram_bins,)


@app.cell
def _(duckdb_conn):
    def alt_ids_to_logged_metrics(alt_ids):
        logged_metrics = userfm.plots.get_logged_metrics(duckdb_conn, alt_ids).pl()
        return duckdb_conn.sql("""
        select
            *
        from (
            unpivot logged_metrics
            on columns('sample_constraint_[0-9]')
            into
                name batch_index
                value constraint_value
        ) join alt_ids using (alt_id)
        """).pl()

    return (alt_ids_to_logged_metrics,)


@app.cell
def _(duckdb_conn):
    exp_rows_rpcn = mo.sql(
        f"""
        select
            alt_id,
            RpCN.*,
            Dataset.sa_inheritance as dataset_name,
        from Conf
        join TrainingAndEvaluation on Conf.id = TrainingAndEvaluation.id
        join Steering on Steering.id = TrainingAndEvaluation.Steering
        join RpCN on Steering.id = RpCN.id
        join Dataset on TrainingAndEvaluation.Dataset = Dataset.id
        join DynamicalSystem on Dataset.id = DynamicalSystem.id
        where true
        and chain_count in (2, 4, 6, 8)
        and pcn_update_count in (1, 2, 4, 8)
        """,
        engine=duckdb_conn
    )
    return (exp_rows_rpcn,)


@app.cell
def _(alt_ids_to_logged_metrics, exp_rows_rpcn):
    to_plot = alt_ids_to_logged_metrics(exp_rows_rpcn)
    return (to_plot,)


@app.cell
def _(
    dataset_constraint_values,
    dataset_histogram_bins,
    exp_rows_rpcn,
    to_plot,
):
    def get_kl_divergences(to_plot):
        kl_divergences = []
        for _dataset_name in dataset_constraint_values['dataset_name'].unique():
            _bins = dataset_histogram_bins[_dataset_name]
            _ds = dataset_constraint_values.filter(dataset_name=_dataset_name)
            ground_truth_hist = np.histogram(_ds.filter(pl.col('constraint_value') > 0)['constraint_value'], bins=_bins)[0].astype(np.float64)
            ground_truth_hist += 1e-8
            ground_truth_hist = ground_truth_hist / ground_truth_hist.sum()
            for _alt_id in to_plot.filter(dataset_name=_dataset_name)['alt_id'].unique():
                steering_hist = np.histogram(to_plot.filter(alt_id=_alt_id)['constraint_value'], bins=_bins)[0].astype(np.float64)
                steering_hist += 1e-8
                steering_hist = steering_hist / steering_hist.sum()
                _kl_divergence = (ground_truth_hist * (np.log(ground_truth_hist) - np.log(steering_hist))).sum()
                kl_divergences.append(dict(dataset_name=_dataset_name, alt_id=_alt_id, kl_divergence=_kl_divergence.sum()))
        kl_divergences = pl.DataFrame(kl_divergences)
        return kl_divergences
    kl_divergences = get_kl_divergences(to_plot).join(exp_rows_rpcn, on='alt_id').sort('alt_id')
    kl_divergences
    return (get_kl_divergences,)


@app.cell
def _(duckdb_conn):
    pivoted = mo.sql(
        f"""
        -- select
        --     case dataset_name
        --     	when 'Lorenz63' then 'L''63'
        --     	when 'FitzHughNagumo' then 'FHN'
        --     end as dataset_name,
        --     * exclude dataset_name
        -- from (
        --     pivot kl_divergences
        --     on chain_count
        --     using first(kl_divergence)
        --     group by dataset_name
        --     order by dataset_name desc
        -- )
        select
            *
        from (
            pivot (select * from to_plot where dataset_name = 'Lorenz63')
            on pcn_update_count
            using mean(cast(constraint_value > 0 as float))
            group by chain_count
            order by chain_count desc
        )
        """,
        engine=duckdb_conn
    )
    return (pivoted,)


@app.cell
def _(pivoted):
    # pivoted_pd = pivoted.to_pandas().set_index("dataset_name")
    # pivoted_pd.index = pivoted_pd.index.rename("Dataset")
    pivoted_pd = pivoted.to_pandas().set_index("chain_count")
    pivoted_pd.columns = pivoted_pd.columns.astype(int).rename("$N$")
    pivoted_pd = pivoted_pd.sort_index(axis=1)
    pivoted_pd
    return (pivoted_pd,)


@app.cell
def _(SAVE_DIR, pivoted_pd):
    _fig_width = page_width / 3
    _fig, _ax = plt.subplots(figsize=(_fig_width, _fig_width / golden_ratio / 1.0))
    (
        sns.heatmap(
            data=pivoted_pd,
            vmin=0,
            # vmax=1,
            annot=True,
            ax=_ax,
            fmt='.2f',
            cmap='viridis',
            cbar_kws=dict(shrink=.85, aspect=7),
        )
        .set(ylabel='KL')
    )
    _ax.xaxis.tick_top()
    xticklabels = _ax.get_xticklabels()
    xticklabels[0].set_text(f'$N$={xticklabels[0].get_text()}')
    _ax.set_xticklabels(xticklabels)
    _ax.xaxis.set_label_text('')
    _ax.xaxis.set_label_position('top')
    _ax.get_figure().savefig(SAVE_DIR/'Heatmap.TV.USMC.pdf', format='pdf', bbox_inches='tight', pad_inches=.00)
    _ax
    return


@app.cell
def _(
    SAVE_DIR,
    alt_ids_to_logged_metrics,
    dataset_constraint_values,
    dataset_histogram_bins,
    exp_rows_rpcn,
    get_kl_divergences,
):
    # _fig_width = page_width * (1 - 1 / 2.45) / 2
    _fig_width = (page_width - 20*pt) / 4
    # _fig, _ax = plt.subplots(figsize=(_fig_width, _fig_width / golden_ratio / 3.0))
    _exp_rows = pl.DataFrame([
        dict(dataset_name='Lorenz63', model='FKSteering', alt_id='4574bsrr'),
        dict(dataset_name='FitzHughNagumo', model='FKSteering', alt_id='njucaaba'),
        dict(dataset_name='Lorenz63', model='SPT', alt_id='7xd4aup5'),
        dict(dataset_name='FitzHughNagumo', model='SPT', alt_id='5utbygnl'),
        dict(dataset_name='Lorenz63', model='RpCN', alt_id=exp_rows_rpcn.filter(dataset_name='Lorenz63', chain_count=8, pcn_update_count=8)['alt_id'][0]),
        dict(dataset_name='FitzHughNagumo', model='RpCN', alt_id=exp_rows_rpcn.filter(dataset_name='FitzHughNagumo', chain_count=8, pcn_update_count=8)['alt_id'][0]),
    ])
    _to_plot = alt_ids_to_logged_metrics(_exp_rows)
    histogram_kl_divergences = get_kl_divergences(_to_plot).join(_exp_rows, on='alt_id')
    # exp_rows = _to_plot
    _plots = []
    for _dataset_name in dataset_constraint_values['dataset_name'].unique():
        _common_kws = dict(
            stat='density',
            x='constraint_value',
            bins=dataset_histogram_bins[_dataset_name],
            element='step',
            linewidth=.5,
        )
        if len(_to_plot.filter(dataset_name=_dataset_name)) == 0:
            continue
        _to_plot_filtered = _to_plot.filter(dataset_name=_dataset_name)
        _plot = (
            sns.displot(
                data=_to_plot_filtered,
                **_common_kws,
                col='model',
                hue='model',
                palette={
                    'FKSteering': 'tab:blue',
                    'RpCN': 'tab:orange',
                },
                common_norm=False,
                alpha=.50,
                legend=False,
                facet_kws=dict(
                    # sharex=False,
                ),
            )
            .set_titles('')
            .set(
                yscale='log'#if _dataset_name == 'FitzHughNagumo' else 'linear',
            )
        )
        for (_row, _col, _hue), _ in _plot.facet_data():
            _ax = _plot.axes[_row][_col]
            _ax.axvline(x=0, c='r', ls=':', lw=2, zorder=-1)
            _ax.minorticks_off()
        _plot.map(
            sns.histplot,
            data=dataset_constraint_values.filter((pl.col('constraint_value') > 0), dataset_name=_dataset_name),
            color='tab:grey',
            **_common_kws,
            label='Dormand-Prince',
            alpha=1,
            zorder=-1,
        ).set_xlabels('').set_ylabels('')
        import matplotlib.patches as mpatches
        custom_legend_items = {
            'Dormand-Prince': mpatches.Patch(color='tab:gray', alpha=1),
            'FK Steering': mpatches.Patch(color='tab:blue', alpha=0.5),
            'RpCN': mpatches.Patch(color='tab:orange', alpha=0.5),
        }
        _plot.add_legend(legend_data=custom_legend_items, title='')
        sns.move_legend(
            _plot, 
            loc='upper center',
            ncol=len(custom_legend_items),
            bbox_to_anchor=(.5, 1.35),
            borderpad=0,
            # frameon=True,
            # fancybox=True,
        )
        _plot.despine(top=False, right=False)
        _plot.figure.set_figwidth(_fig_width * len(_to_plot_filtered['alt_id'].unique()))#/ len(dataset_constraint_values['dataset_name'].unique()))
        _plot.figure.set_figheight(_fig_width / golden_ratio / 1)
        # h_pad, w_pad = .2, .5
        # w_pad_pt = w_pad*plt.rcParams['axes.labelsize']*pt
        # h_pad_pt = h_pad*plt.rcParams['axes.labelsize']*pt
        # _plot.tight_layout()
        # _plot.tight_layout(h_pad=h_pad, w_pad=w_pad)
        # _plot.figure.subplots_adjust(left=0.18)
        _plots.append(_plot)
        userfm.plots.save_all_subfigures(_plot, SAVE_DIR/_dataset_name/f'NiceHistogram')
        # userfm.plots.save_all_subfigures(_plot, f'Histogram.{_dataset_name}')
    mo.hstack(_plots)
    return (histogram_kl_divergences,)


@app.cell
def _(histogram_kl_divergences):
    histogram_kl_divergences.write_csv('hist.csv')
    return


if __name__ == "__main__":
    app.run()
