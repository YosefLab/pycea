from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import treedata as td
from matplotlib.axes import Axes

from ._legend import _categorical_legend, _render_legends
from ._utils import _get_categorical_colors


def n_extant(
    tdata: td.TreeData,
    color: Sequence[str] | str | None = None,
    *,
    data: pd.DataFrame | None = None,
    key: str = "n_extant",
    depth_key: str | None = None,
    n_extant_key: str | None = None,
    stat: Literal["count", "proportion", "percent"] = "count",
    order: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
    na_color: str | None = "lightgray",
    legend: bool | None = None,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Plot extant branches over time.

    Parameters
    ----------
    tdata
        The TreeData object.
    color
        Column(s) in `data` to color by. Determined from `data` when `None`.
    data
        Extant counts to plot. Uses `tdata.uns[key]` if `None`.
    key
        Key in `tdata.uns` storing extant counts when `data` is `None`.
    depth_key
        Column storing time or depth values. If `None`, uses the first column in `data`.
    n_extant_key
        Column storing extant counts. If `None`, uses the second column in `data`.
    stat
        Statistic to compute for the ribbons: 'count', 'fraction', or 'percent'.
    order
        Order of group categories in the stack.
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …), a :class:`~cycler.Cycler` object, a dict mapping
        categories to colors, or a sequence of colors. Colors must be valid to
        matplotlib. (see :func:`~matplotlib.colors.is_color_like`).
        If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
        variable already has colors stored in `tdata.uns["{var}_colors"]`.
        If provided, values of `tdata.uns["{var}_colors"]` will be set.
    na_color
        The color to use for annotations with missing data.
    legend
        Whether to add a legend to the plot. By default, a legend is added if there
        are <= 20 distinct categories.
    ax
        Axes on which to draw the plot. Creates new axes when `None`.
    legend_kwargs
        Additional keyword arguments for the legend.

    Returns
    -------
    ax - Axis containing the ribbon plot.
    """
    df = data if data is not None else tdata.uns.get(key)
    if df is None:
        raise KeyError(f"{key!r} not found in tdata.uns and no data provided")
    df = df.copy()

    if depth_key is None:
        depth_key = df.columns[0]
    if n_extant_key is None:
        n_extant_key = df.columns[1]
    if color is None:
        other_cols = [c for c in df.columns if c not in {depth_key, n_extant_key, "tree"}]
        if len(other_cols) == 1:
            color = other_cols[0]
        elif len(other_cols) > 1:
            color = other_cols

    if na_color is not None:
        for col in df.columns:
            if df[col].dtype.name == "category":
                df[col] = df[col].cat.add_categories("NA").fillna("NA")

    if isinstance(color, Sequence) and not isinstance(color, str):
        df["_group"] = df[list(color)].astype(str).agg("_".join, axis=1)
        legend_title = "_".join(color)
        df["_group"] = df["_group"].astype("category")
    elif color is not None:
        df["_group"] = df[color]
        legend_title = str(color)
    else:
        df["_group"] = "_all"
        legend_title = "_all"

    pivot = df.pivot_table(
        index=depth_key, columns="_group", values=n_extant_key, aggfunc="sum", fill_value=0, observed=True
    ).sort_index()
    if order is not None:
        pivot = pivot[[c for c in order if c in pivot.columns]]
    cumcount = pivot.cumsum(axis=1)

    if stat != "count":
        total = pivot.sum(axis=1)
        if stat == "proportion":
            cumcount = cumcount.div(total, axis=0)
        elif stat == "percent":
            cumcount = cumcount.div(total, axis=0) * 100
        else:
            raise ValueError("Invalid stat. Choose from 'count', 'proportion', or 'percent'.")

    if ax is None:
        _, ax = plt.subplots()
    ax = cast(Axes, ax)

    color_key = legend_title
    color_map = _get_categorical_colors(
        tdata,
        color_key,
        palette=palette,
        data=df["_group"].cat.remove_categories("NA") if df["_group"].dtype.name == "category" else df["_group"],
    )
    if (na_color is not None) and ("NA" in df["_group"].values):
        color_map["NA"] = na_color
    legends: list[dict[str, Any]] = []

    for i, group in enumerate(cumcount.columns):
        lower = cumcount.iloc[:, i - 1] if i > 0 else np.zeros(cumcount.shape[0])
        upper = cumcount.iloc[:, i]
        ax.fill_between(cumcount.index, lower, upper, color=color_map.get(group))

    ax.set_xlabel(depth_key)
    ax.set_ylabel(stat)
    ax.margins(x=0, y=0)

    if color is not None:
        legends.append(_categorical_legend(legend_title, color_map))
    if legend is True or (legend is None and legends and len(color_map) <= 20):
        _render_legends(ax, legends, anchor_x=1.01, spacing=0.02, shared_kwargs=legend_kwargs)
    return ax
