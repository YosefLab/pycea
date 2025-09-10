from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import treedata as td
from matplotlib.axes import Axes

from ._legend import _categorical_legend, _render_legends
from ._utils import _get_categorical_colors


def n_extant(
    tdata: td.TreeData,
    group_key: Sequence[str] | str | None = None,
    *,
    data: pd.DataFrame | None = None,
    key: str = "n_extant",
    time_key: str = "time",
    n_extant_key: str = "n_extant",
    stat: Literal["count", "fraction", "percent"] = "count",
    order: Sequence[str] | None = None,
    legend: bool | None = None,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Plot extant branch counts over time.

    Displays the number of extant branches for each depth bin, optionally grouped
    by observation variables.

    Parameters
    ----------
    tdata
        TreeData object.
    group_key
        Column(s) in `data` identifying groups. Determined from `data` when `None`.
    data
        Extant counts to plot. Uses ``tdata.uns[key]`` if ``None``.
    key
        Key in ``tdata.uns`` storing extant counts when ``data`` is ``None``.
    time_key
        Column storing time or depth values.
    n_extant_key
        Column storing extant counts.
    stat
        Statistic to compute for the ribbons: 'count', 'fraction', or 'percent'.
    order
        Order of group categories in the stack.
    legend
        Whether to add a legend.
    ax
        Axes on which to draw the plot. Creates new axes when ``None``.
    legend_kwargs
        Additional keyword arguments for the legend.

    Returns
    -------
    ax - Axis containing the ribbon plot.
    """
    df = data.copy() if data is not None else tdata.uns.get(key)
    if df is None:
        raise KeyError(f"{key!r} not found in tdata.uns and no data provided")
    df = df.copy()

    if group_key is None:
        other_cols = [c for c in df.columns if c not in {time_key, n_extant_key}]
        if len(other_cols) == 1:
            group_key = other_cols[0]
        elif len(other_cols) > 1:
            group_key = other_cols

    if isinstance(group_key, Sequence) and not isinstance(group_key, str):
        df["_group"] = df[list(group_key)].astype(str).agg("_".join, axis=1)
        legend_title = "_".join(group_key)
    elif group_key is not None:
        df["_group"] = df[group_key]
        legend_title = str(group_key)
    else:
        df["_group"] = "_all"
        legend_title = "_all"

    pivot = df.pivot_table(
        index=time_key, columns="_group", values=n_extant_key, aggfunc="sum", fill_value=0
    ).sort_index()
    if order is not None:
        pivot = pivot[[c for c in order if c in pivot.columns]]
    cumcount = pivot.cumsum(axis=1)

    if stat != "count":
        total = pivot.sum(axis=1)
        if stat == "fraction":
            cumcount = cumcount.div(total, axis=0)
        elif stat == "percent":
            cumcount = cumcount.div(total, axis=0) * 100
        else:
            raise ValueError("Invalid stat. Choose from 'count', 'fraction', or 'percent'.")

    if ax is None:
        _, ax = plt.subplots()
    ax = cast(Axes, ax)

    color_key = legend_title
    color_map = _get_categorical_colors(tdata, color_key, df["_group"], save=False)
    legends: list[dict[str, Any]] = []

    for i, group in enumerate(cumcount.columns):
        lower = cumcount.iloc[:, i - 1] if i > 0 else np.zeros(cumcount.shape[0])
        upper = cumcount.iloc[:, i]
        ax.fill_between(cumcount.index, lower, upper, color=color_map.get(group))

    ax.set_xlabel(time_key)
    ylabel = n_extant_key if stat == "count" else f"{stat} of {n_extant_key}"
    ax.set_ylabel(ylabel)

    if group_key is not None:
        legends.append(_categorical_legend(legend_title, color_map))
    if legend is True or (legend is None and legends):
        _render_legends(ax, legends, anchor_x=1.01, spacing=0.02, shared_kwargs=legend_kwargs)
    return ax
