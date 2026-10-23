import copy
from pathlib import Path

import polars as pl


def save_legend(plot, plot_name, format='pdf'):
    plot.figure.savefig(
        f'{plot_name}.legend.{format}', format=format,
        bbox_inches=plot._legend.get_window_extent().transformed(plot.figure.dpi_scale_trans.inverted()).expanded(1.007, 1.05)
    )


def save_all_subfigures(plot, plot_name, format='pdf', renaming=None, metadata_dataframe=None):
    if metadata_dataframe is not None:
        metadata_dataframe.write_csv(f'{plot_name}.csv')
    renaming = renaming or {}
    p = copy.deepcopy(plot)
    if p._legend is not None:
        save_legend(p, plot_name)

        # save each subplot
        p._legend.set_visible(False)
    for (row, col, hue), data in p.facet_data():
        pp = copy.deepcopy(p)
        # ax = p.axes[row][col]
        for (r, c, h), d in pp.facet_data():
            if r != row or c != col:
                ax_other = pp.axes[r][c]
                ax_other.remove()
        variable_names = []
        if len(p.row_names) > 0:
            variable_names.append(renaming.get(p.row_names[row], p.row_names[row]))
        if len(p.col_names) > 0:
            variable_names.append(renaming.get(p.col_names[col], p.col_names[col]))
        if len(variable_names) > 0:
            save_name = f'{plot_name}.{"__".join(variable_names)}.{format}'
        else:
            save_name = f'{plot_name}.{format}'
        # pp.fig.set_layout_engine(layout="constrained",  h_pad=0.00, w_pad=0.00)
        pp.savefig(save_name, format=format)


def get_logged_metrics_file_paths(connection, conf_rows, file_path_format='~/out/rpcn-camera-ready/runs/{}/metrics.csv'):
    paths = connection.sql(f"""
        select format({file_path_format!r}, alt_id) as path from conf_rows
    """).pl()
    exists = []
    for f in paths['path']:
        f = Path(f).expanduser()
        exists.append(f.exists())
    paths = pl.DataFrame(dict(
        path=paths['path'], exists=exists,
    ))
    return connection.sql('select * from paths')


def get_logged_metrics(connection, alt_ids, file_paths=None):
    if file_paths is None:
        file_paths = get_logged_metrics_file_paths(connection, alt_ids)
    connection.sql("""
    set variable dataset_metrics_filepaths = (
        select list(path) from file_paths where exists
    )
    """)
    logged_metrics = connection.sql("""
    select
        split(filename, '/')[-2] as alt_id,
        * exclude(filename),
    from read_csv(getvariable(dataset_metrics_filepaths), filename=true, union_by_name=true)
    """)
    return logged_metrics
