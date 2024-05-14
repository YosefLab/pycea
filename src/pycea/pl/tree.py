from __future__ import annotations

from collections.abc import Mapping, Sequence

import cycler
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import treedata as td
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from pycea._utils import _get_keyed_edge_data

from ._utils import (
    _get_categorical_colors,
    layout_tree,
)


def branches(
    tdata: td.TreeData,
    key: str = None,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    color: str = "black",
    color_na: str = "lightgrey",
    linewidth: int | float | str = 1,
    linewidth_na: int | float = 1,
    cmap: str | mcolors.Colormap = "viridis",
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    ax: Axes | None = None,
    **kwargs,
):
    """Plot the branches of a tree.

    Parameters
    ----------
    tdata
        The `td.TreeData` object.
    key
        The `obst` key of the tree to plot.
    polar
        Whether to plot the tree in polar coordinates.
    extend_branches
        Whether to extend branches so the tips are at the same depth.
    angled_branches
        Whether to plot branches at an angle.
    color
        Either a color name, or a key for an attribute of the edges to color by.
    color_na
        The color to use for edges with missing data.
    linewidth
        Either an numeric width, or a key for an attribute of the edges to set the linewidth.
    linewidth_na
        The linewidth to use for edges with missing data.
    {doc_common_plot_args}
    kwargs
        Additional keyword arguments passed to `matplotlib.collections.LineCollection`.

    Returns
    -------
    `matplotlib.axes.Axes`
    """
    kwargs = kwargs if kwargs else {}
    tree = tdata.obst[key]

    # Get layout
    node_coords, branch_coords, leaves, depth = layout_tree(
        tree, polar=polar, extend_branches=extend_branches, angled_branches=angled_branches
    )
    segments = []
    edges = []
    for edge, (lat, lon) in branch_coords.items():
        coords = np.array([lon, lat] if polar else [lat, lon]).T
        segments.append(coords)
        edges.append(edge)
    kwargs.update({"segments": segments})
    # Get colors
    if mcolors.is_color_like(color):
        kwargs.update({"color": color})
    elif isinstance(color, str):
        color_data = _get_keyed_edge_data(tree, color)
        if color_data.dtype.kind in ["i", "f"]:
            norm = plt.Normalize(vmin=color_data.min(), vmax=color_data.max())
            cmap = plt.get_cmap(cmap)
            colors = [cmap(norm(color_data[edge])) if edge in color_data.index else color_na for edge in edges]
            kwargs.update({"color": colors})
        else:
            cmap = _get_categorical_colors(tdata, color, color_data, palette)
            colors = [cmap[color_data[edge]] if edge in color_data.index else color_na for edge in edges]
            kwargs.update({"color": colors})
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the edges.")
    # Get linewidths
    if isinstance(linewidth, (int, float)):
        kwargs.update({"linewidth": linewidth})
    elif isinstance(linewidth, str):
        linewidth_data = _get_keyed_edge_data(tree, linewidth)
        if linewidth_data.dtype.kind in ["i", "f"]:
            linewidths = [linewidth_data[edge] if edge in linewidth_data.index else linewidth_na for edge in edges]
            kwargs.update({"linewidth": linewidths})
        else:
            raise ValueError("Invalid linewidth data type. Edge attribute must be int or float")
    else:
        raise ValueError("Invalid linewidth value. Must be int, float, or an str specifying an attribute of the edges.")
    # Plot
    if not ax:
        subplot_kw = {"projection": "polar"} if polar else None
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    elif (ax.name == "polar") != polar:
        raise ValueError("Provided axis does not match the requested 'polar' setting.")
    ax.add_collection(LineCollection(**kwargs))
    # Configure plot
    lat_lim = (-0.1, depth)
    lon_lim = (0, 2 * np.pi)
    ax.set_xlim(lon_lim if polar else lat_lim)
    ax.set_ylim(lat_lim if polar else lon_lim)
    ax.axis("off")
    ax._attrs = {"node_coords": node_coords, "leaves": leaves, "depth": depth, "offset": depth, "polar": polar}
    return ax
