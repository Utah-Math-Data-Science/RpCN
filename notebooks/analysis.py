import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import duckdb
    import seaborn as sns
    sns.set_theme(style="white", font_scale=2.6)
    import matplotlib.pyplot as plt

    import toy.plots

    return mo, plt, sns, toy


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        attach if not exists 'runs.sqlite';
        use runs;
        """
    )
    return


@app.cell
def _(conf, mo, rpcn):
    rpcn_rows = mo.sql(
        f"""
        select
            Conf.alt_id,
            RpCN.n_chains, RpCN.pcn_update_count,
        from Conf
        join RpCN on Conf.SteeringBase = RpCN.id
        where true
        and RpCN.inverse_temperature = 25
        and 1 <= RpCN.n_chains and RpCN.n_chains <= 8
        and 1 <= RpCN.pcn_update_count and RpCN.pcn_update_count <= 32
        """
    )
    return (rpcn_rows,)


@app.cell
def _(rpcn_rows, toy):
    logged_metrics = toy.plots.get_logged_metrics(rpcn_rows)
    logged_metrics
    return (logged_metrics,)


@app.cell(hide_code=True)
def _(logged_metrics, mo, rpcn_rows):
    to_plot = mo.sql(
        f"""
        select
        	alt_id,
            n_chains, pcn_update_count,
            mean(reward),
            max(reward),
        from rpcn_rows
        join logged_metrics using (alt_id)
        group by alt_id, n_chains, pcn_update_count
        """
    )
    return (to_plot,)


@app.cell(hide_code=True)
def _(mo, to_plot):
    pivoted = mo.sql(
        f"""
        pivot (
            select
            	n_chains, pcn_update_count,
                "mean(reward)",
            from to_plot
            where true
            --and 1 <= n_chains and n_chains <= 7
            and n_chains in (1, 2, 4, 8)
            and pcn_update_count in (1, 2, 4, 8, 16, 32)
        ) 
        on pcn_update_count 
        using first("mean(reward)") 
        group by n_chains
        """
    )
    return (pivoted,)


@app.cell
def _(pivoted):
    pivoted_pd = pivoted.to_pandas().set_index('n_chains')
    pivoted_pd.index = pivoted_pd.index.rename('$K+1$, replicas')
    pivoted_pd.columns = pivoted_pd.columns.astype(int).rename('$M$, pCN steps with replica exchanges')
    pivoted_pd = pivoted_pd.sort_index().sort_index(axis=1)
    pivoted_pd
    return (pivoted_pd,)


@app.cell
def _(pivoted_pd, plt, sns):
    fig, ax = plt.subplots(figsize=(11.5,5))
    (
        sns.heatmap(
            data=pivoted_pd,
            vmax=1,
            annot=True,
            vmin=-2,
            ax=ax,
            fmt='.2f',
            cmap='viridis'
        )
    )
    ax.invert_yaxis()
    ax
    return (ax,)


@app.cell
def _(ax):
    ax.get_figure().savefig('Heatmap.RpCN.pdf', format='pdf', bbox_inches='tight', pad_inches=.06)
    return


if __name__ == "__main__":
    app.run()
