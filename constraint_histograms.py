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
    SAVE_DIR = Path('~/Overleaf/RpCN/imgs/dynamical_systems').expanduser()
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
    def alt_ids_to_logged_metrics(alt_ids, has_constraint_values=True):
        logged_metrics = userfm.plots.get_logged_metrics(duckdb_conn, alt_ids).pl()
        if not has_constraint_values:
            return logged_metrics
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
        and log_constraint_values
        and chain_count in (2, 4, 6, 8)
        and pcn_update_count in (1, 2, 4)
        """,
        engine=duckdb_conn
    )
    return (exp_rows_rpcn,)


@app.cell
def _(duckdb_conn):
    exp_rows_fk = mo.sql(
        f"""
        select
            alt_id,
            FeynmanKac.*,
            Dataset.sa_inheritance as dataset_name,
        from Conf
        join TrainingAndEvaluation on Conf.id = TrainingAndEvaluation.id
        join Steering on Steering.id = TrainingAndEvaluation.Steering
        join FeynmanKac on Steering.id = FeynmanKac.id
        join Dataset on TrainingAndEvaluation.Dataset = Dataset.id
        join DynamicalSystem on Dataset.id = DynamicalSystem.id
        where true
        and symmetric_binary_reward
        and penalization_power = 0
        and not log_constraint_values
        and tilt in (select tilt from SweepDynamicalSystem)
        and chain_count = 1
        and ensemble_size in (2, 4, 6, 8)
        and potential in ('DIFFERENCE', 'MAX', 'SUM')
        order by dataset_name, ensemble_size, potential
        """,
        engine=duckdb_conn
    )
    return (exp_rows_fk,)


@app.cell
def _(duckdb_conn):
    exp_rows_crepe = mo.sql(
        f"""
        select
            alt_id,
            CREPE.*,
            Dataset.sa_inheritance as dataset_name,
        from Conf
        join TrainingAndEvaluation on Conf.id = TrainingAndEvaluation.id
        join Steering on Steering.id = TrainingAndEvaluation.Steering
        join CREPE on Steering.id = CREPE.id
        join Dataset on TrainingAndEvaluation.Dataset = Dataset.id
        join DynamicalSystem on Dataset.id = DynamicalSystem.id
        where true
        and symmetric_binary_reward
        --and not log_constraint_values
        and tilt in (select tilt from SweepDynamicalSystem)
        and level_count in (2, 4, 6, 8)
        and block_count in (4, 8, 16)
        order by dataset_name, level_count, block_count
        """,
        engine=duckdb_conn
    )
    return (exp_rows_crepe,)


@app.cell
def _(alt_ids_to_logged_metrics, exp_rows_crepe, exp_rows_fk, exp_rows_rpcn):
    to_plot = alt_ids_to_logged_metrics(exp_rows_rpcn)
    to_plot_fk = alt_ids_to_logged_metrics(exp_rows_fk, has_constraint_values=False).join(exp_rows_fk, on='alt_id').sort('alt_id')
    to_plot_crepe = alt_ids_to_logged_metrics(exp_rows_crepe, has_constraint_values=True).join(exp_rows_crepe, on='alt_id').sort('alt_id')
    return


@app.cell
def _(dataset_constraint_values, dataset_histogram_bins):
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
    # kl_divergences = get_kl_divergences(to_plot).join(exp_rows_rpcn, on='alt_id').sort('alt_id')
    # kl_divergences
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
            pivot to_plot
            on pcn_update_count
            using mean(satisfactory_sample_count) / 500
            group by dataset_name, chain_count
            order by chain_count desc
        )
        """,
        engine=duckdb_conn
    )
    return (pivoted,)


@app.cell
def _(duckdb_conn):
    pivoted_fk = mo.sql(
        f"""
        select
            *
        from (
            pivot to_plot_fk
            on case potential
            	when 'DIFFERENCE' then 'DIFF'
            	else potential
            end
            using mean(satisfactory_sample_count) / 500
            group by dataset_name, ensemble_size
            order by ensemble_size desc
        )
        """,
        engine=duckdb_conn
    )
    return (pivoted_fk,)


@app.cell
def _(duckdb_conn):
    pivoted_crepe = mo.sql(
        f"""
        select
            *
        from (
            pivot to_plot_crepe
            on block_count
            using mean(satisfactory_sample_count) / 500
            group by dataset_name, level_count
            order by level_count desc
        )
        """,
        engine=duckdb_conn
    )
    return (pivoted_crepe,)


@app.function
def pivoted_to_pandas(pivoted, index, columns_name):
    pivoted_pd = pivoted.to_pandas().set_index(index)
    pivoted_pd.columns = pivoted_pd.columns.rename(columns_name)
    pivoted_pd = pivoted_pd.sort_index(axis=1)
    # pivot column names are strings, so sort numerical ones by value (e.g., '16' after '8')
    numerical_columns = [c for c in pivoted_pd.columns if str(c).lstrip('-').replace('.', '', 1).isdigit()]
    other_columns = [c for c in pivoted_pd.columns if c not in numerical_columns]
    return pivoted_pd[other_columns + sorted(numerical_columns, key=float)]


@app.function
def plot_heatmaps(pivoted_pd, ylabel, save_dir, model_name):
    fig_width = page_width / 3
    axes = []
    for dataset_name in pivoted_pd['dataset_name'].unique():
        fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden_ratio / 1.0))
        fig.set_layout_engine(layout="constrained",  h_pad=0.01, w_pad=0.01)
        (
            sns.heatmap(
                data=pivoted_pd[pivoted_pd['dataset_name'] == dataset_name].drop(columns='dataset_name'),
                vmin=0,
                vmax=1,
                annot=True,
                ax=ax,
                fmt='.2f',
                cbar=False,
                cmap='viridis',
                cbar_kws=dict(shrink=.85, aspect=7),
            )
            .set(ylabel=ylabel)
        )
        fig.savefig(save_dir/'ablation'/f'Heatmap.SR.{model_name}.{dataset_name}.pdf', format='pdf')
        axes.append(ax)
    return axes


@app.cell
def _(pivoted):
    pivoted_pd = pivoted_to_pandas(pivoted, index='chain_count', columns_name='$M$')
    pivoted_pd
    return (pivoted_pd,)


@app.cell
def _(pivoted_fk):
    pivoted_pd_fk = pivoted_to_pandas(pivoted_fk, index='ensemble_size', columns_name='Potential')
    pivoted_pd_fk
    return (pivoted_pd_fk,)


@app.cell
def _(pivoted_crepe):
    pivoted_pd_crepe = pivoted_to_pandas(pivoted_crepe, index='level_count', columns_name='Even-odd swaps')
    pivoted_pd_crepe
    return (pivoted_pd_crepe,)


@app.cell
def _(SAVE_DIR, pivoted_pd):
    _axes = plot_heatmaps(pivoted_pd, ylabel='$K+1$', save_dir=SAVE_DIR, model_name='RpCN')
    mo.hstack(_axes)
    return


@app.cell
def _(SAVE_DIR, pivoted_pd_fk):
    _axes = plot_heatmaps(pivoted_pd_fk, ylabel='Ensemble size', save_dir=SAVE_DIR, model_name='FKSteering')
    mo.hstack(_axes)
    return


@app.cell
def _(SAVE_DIR, pivoted_pd_crepe):

    _axes = plot_heatmaps(pivoted_pd_crepe, ylabel='Marginals', save_dir=SAVE_DIR, model_name='CREPE')
    mo.hstack(_axes)
    return


@app.cell
def _(
    SAVE_DIR,
    alt_ids_to_logged_metrics,
    dataset_constraint_values,
    dataset_histogram_bins,
    exp_rows_crepe,
    exp_rows_rpcn,
    get_kl_divergences,
):
    # _fig_width = page_width * (1 - 1 / 2.45) / 2
    _fig_width = page_width / 3
    # _fig, _ax = plt.subplots(figsize=(_fig_width, _fig_width / golden_ratio / 3.0))
    _exp_rows = pl.DataFrame([
        dict(dataset_name='Lorenz63', model='FKSteering', alt_id='4574bsrr'),
        dict(dataset_name='FitzHughNagumo', model='FKSteering', alt_id='apxxdxcd'),
        # dict(dataset_name='Lorenz63', model='SPT', alt_id='7xd4aup5'),
        # dict(dataset_name='FitzHughNagumo', model='SPT', alt_id='5utbygnl'),
        dict(dataset_name='Lorenz63', model='CREPE', alt_id=exp_rows_crepe.filter(dataset_name='Lorenz63', level_count=6, block_count=16)['alt_id'][0]),
        dict(dataset_name='FitzHughNagumo', model='CREPE', alt_id=exp_rows_crepe.filter(dataset_name='FitzHughNagumo', level_count=2, block_count=16)['alt_id'][0]),
        dict(dataset_name='Lorenz63', model='RpCN', alt_id=exp_rows_rpcn.filter(dataset_name='Lorenz63', chain_count=8, pcn_update_count=4)['alt_id'][0]),
        dict(dataset_name='FitzHughNagumo', model='RpCN', alt_id=exp_rows_rpcn.filter(dataset_name='FitzHughNagumo', chain_count=8, pcn_update_count=4)['alt_id'][0]),
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
                col_order=['FKSteering', 'CREPE', 'RpCN'],
                hue='model',
                palette={
                    'FKSteering': 'tab:blue',
                    'CREPE': 'tab:green',
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
            'CREPE': mpatches.Patch(color='tab:green', alpha=0.5),
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
        # _plot.fig.set_layout_engine(layout="constrained",  h_pad=0.00, w_pad=0.00)
        # h_pad, w_pad = .2, .5
        # w_pad_pt = w_pad*plt.rcParams['axes.labelsize']*pt
        # h_pad_pt = h_pad*plt.rcParams['axes.labelsize']*pt
        # _plot.tight_layout()
        # _plot.tight_layout(h_pad=h_pad, w_pad=w_pad)
        # _plot.figure.subplots_adjust(left=0.18)
        _plots.append(_plot)
        userfm.plots.save_all_subfigures(_plot, SAVE_DIR/'ConstraintHistogram'/_dataset_name/f'NiceHistogram')
        # userfm.plots.save_all_subfigures(_plot, f'Histogram.{_dataset_name}')
    mo.vstack(_plots)
    return (histogram_kl_divergences,)


@app.cell
def _(histogram_kl_divergences):
    histogram_kl_divergences.write_csv('hist.csv')
    histogram_kl_divergences
    return


if __name__ == "__main__":
    app.run()
