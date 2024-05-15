from __future__ import annotations

from collections.abc import Mapping, Sequence

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import treedata as td
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from pycea.utils import get_keyed_edge_data, get_keyed_node_data, get_keyed_obs_data

from ._docs import _doc_params, doc_common_plot_args
from ._utils import (
    _get_categorical_colors,
    _get_categorical_markers,
    _series_to_rgb_array,
    layout_tree,
)


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def branches(
    tdata: td.TreeData,
    key: str = None,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    color: str = "black",
    linewidth: int | float | str = 1,
    cmap: str | mcolors.Colormap = "viridis",
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    na_color: str = "lightgrey",
    na_linewidth: int | float = 1,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot the branches of a tree.

    Parameters
    ----------
    tdata
        The `treedata.TreeData` object.
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
    linewidth
        Either an numeric width, or a key for an attribute of the edges to set the linewidth.
    {common_plot_args}
    na_color
        The color to use for edges with missing data.
    na_linewidth
        The linewidth to use for edges with missing data.
    kwargs
        Additional keyword arguments passed to `matplotlib.collections.LineCollection`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    kwargs = kwargs if kwargs else {}
    if not key:
        key = next(iter(tdata.obst.keys()))
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
        color_data = get_keyed_edge_data(tree, color)
        if color_data.dtype.kind in ["i", "f"]:
            norm = plt.Normalize(vmin=color_data.min(), vmax=color_data.max())
            cmap = plt.get_cmap(cmap)
            colors = [cmap(norm(color_data[edge])) if edge in color_data.index else na_color for edge in edges]
            kwargs.update({"color": colors})
        else:
            cmap = _get_categorical_colors(tdata, color, color_data, palette)
            colors = [cmap[color_data[edge]] if edge in color_data.index else na_color for edge in edges]
            kwargs.update({"color": colors})
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the edges.")
    # Get linewidths
    if isinstance(linewidth, (int, float)):
        kwargs.update({"linewidth": linewidth})
    elif isinstance(linewidth, str):
        linewidth_data = get_keyed_edge_data(tree, linewidth)
        if linewidth_data.dtype.kind in ["i", "f"]:
            linewidths = [linewidth_data[edge] if edge in linewidth_data.index else na_linewidth for edge in edges]
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
    ax.add_collection(LineCollection(zorder=1, **kwargs))
    # Configure plot
    lat_lim = (-0.2, depth)
    lon_lim = (0, 2 * np.pi)
    ax.set_xlim(lon_lim if polar else lat_lim)
    ax.set_ylim(lat_lim if polar else lon_lim)
    ax.axis("off")
    ax._attrs = {
        "node_coords": node_coords,
        "leaves": leaves,
        "depth": depth,
        "offset": depth,
        "polar": polar,
        "tree_key": key,
    }
    return ax


# For internal use
_branches = branches


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def nodes(
    tdata: td.TreeData,
    nodes: str | Sequence[str] = "internal",
    color: str = "black",
    style: str = "o",
    size: int | float | str = 10,
    cmap: str | mcolors.Colormap = None,
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    markers: Sequence[str] | Mapping[str] = None,
    vmax: int | float | None = None,
    vmin: int | float | None = None,
    na_color: str = "#FFFFFF00",
    na_style: str = "none",
    na_size: int | float = 0,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot the nodes of a tree.

    Parameters
    ----------
    tdata
        The TreeData object.
    nodes
        Either "all", "leaves", "internal", or a list of nodes to plot.
    color
        Either a color name, or a key for an attribute of the nodes to color by.
    style
        Either a marker name, or a key for an attribute of the nodes to set the marker.
        Can be numeric but will always be treated as a categorical variable.
    size
        Either an numeric size, or a key for an attribute of the nodes to set the size.
    {common_plot_args}
    markers
        Object determining how to draw the markers for different levels of the style variable.
        You can pass a list of markers or a dictionary mapping levels of the style variable to markers.
    na_color
        The color to use for annotations with missing data.
    na_style
        The marker to use for annotations with missing data.
    na_size
        The size to use for annotations with missing data.
    kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    kwargs = kwargs if kwargs else {}
    if not ax:
        ax = plt.gca()
    attrs = ax._attrs if hasattr(ax, "_attrs") else None
    if not attrs:
        raise ValueError("Branches most be plotted with pycea.pl.branches before annotations can be plotted.")
    if not cmap:
        cmap = mpl.rcParams["image.cmap"]
    cmap = plt.get_cmap(cmap)
    tree = tdata.obst[attrs["tree_key"]]
    # Get nodes
    all_nodes = list(attrs["node_coords"].keys())
    leaves = list(attrs["leaves"])
    if nodes == "all":
        nodes = all_nodes
    elif nodes == "leaves":
        nodes = leaves
    elif nodes == "internal":
        nodes = [node for node in all_nodes if node not in leaves]
    elif isinstance(nodes, Sequence):
        if set(nodes).issubset(all_nodes):
            nodes = list(nodes)
        else:
            raise ValueError("Nodes must be a list of nodes in the tree.")
    else:
        raise ValueError("Invalid nodes value. Must be 'all', 'leaves', 'no_leaves', or a list of nodes.")
    # Get coordinates
    coords = np.vstack([attrs["node_coords"][node] for node in nodes])
    if attrs["polar"]:
        kwargs.update({"x": coords[:, 1], "y": coords[:, 0]})
    else:
        kwargs.update({"x": coords[:, 0], "y": coords[:, 1]})
    kwargs_list = []
    # Get colors
    if mcolors.is_color_like(color):
        kwargs.update({"color": color})
    elif isinstance(color, str):
        color_data = get_keyed_node_data(tree, color)
        if color_data.dtype.kind in ["i", "f"]:
            if not vmin:
                vmin = color_data.min()
            if not vmax:
                vmax = color_data.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colors = [cmap(norm(color_data[node])) if node in color_data.index else na_color for node in nodes]
            kwargs.update({"color": colors})
        else:
            cmap = _get_categorical_colors(tdata, color, color_data, palette)
            colors = [cmap[color_data[node]] if node in color_data.index else na_color for node in nodes]
            kwargs.update({"color": colors})
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the nodes.")
    # Get sizes
    if isinstance(size, (int, float)):
        kwargs.update({"s": size})
    elif isinstance(size, str):
        size_data = get_keyed_node_data(tree, size)
        sizes = [size_data[node] if node in size_data.index else na_size for node in nodes]
        kwargs.update({"s": sizes})
    else:
        raise ValueError("Invalid size value. Must be int, float, or an str specifying an attribute of the nodes.")
    # Get markers
    if style in mmarkers.MarkerStyle.markers:
        kwargs.update({"marker": style})
    elif isinstance(style, str):
        style_data = get_keyed_node_data(tree, style)
        mmap = _get_categorical_markers(tdata, style, style_data, markers)
        styles = [mmap[style_data[node]] if node in style_data.index else na_style for node in nodes]
        for style in set(styles):
            style_kwargs = {}
            idx = [i for i, x in enumerate(styles) if x == style]
            for key, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    style_kwargs[key] = [value[i] for i in idx]
                else:
                    style_kwargs[key] = value
            style_kwargs.update({"marker": style})
            kwargs_list.append(style_kwargs)
    else:
        raise ValueError("Invalid style value. Must be a marker name, or an str specifying an attribute of the nodes.")
    # Plot
    if len(kwargs_list) > 0:
        for kwargs in kwargs_list:
            ax.scatter(**kwargs)
    else:
        ax.scatter(**kwargs)
    return ax


# For internal use
_nodes = nodes


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def annotation(
    tdata: td.TreeData,
    keys: str | Sequence[str] = None,
    width: int | float = 0.05,
    gap: int | float = 0.03,
    label: bool | str | Sequence[str] = True,
    cmap: str | mcolors.Colormap = None,
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    vmax: int | float | None = None,
    vmin: int | float | None = None,
    na_color: str = "white",
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot leaf annotations.

    Parameters
    ----------
    tdata
        The TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to plot.
    width
        The width of the annotation bar relative to the tree.
    gap
        The gap between the annotation bar and the tree relative to the tree.
    label
        Annotation labels. If `True`, the keys are used as labels.
        If a string or a sequence of strings, the strings are used as labels.
    {common_plot_args}
    na_color
        The color to use for annotations with missing data.
    kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    if not ax:
        ax = plt.gca()
    attrs = ax._attrs if hasattr(ax, "_attrs") else None
    if not attrs:
        raise ValueError("Branches most be plotted with pycea.pl.branches before annotations can be plotted.")
    if not keys:
        raise ValueError("No keys provided. Please provide one or more keys to plot.")
    keys = [keys] if isinstance(keys, str) else keys
    if not cmap:
        cmap = mpl.rcParams["image.cmap"]
    cmap = plt.get_cmap(cmap)
    # Get data
    data, is_array = get_keyed_obs_data(tdata, keys)
    data = data.loc[attrs["leaves"]]
    numeric_data = data.select_dtypes(exclude="category")
    if len(numeric_data) > 0 and not vmin:
        vmin = numeric_data.min().min()
    if len(numeric_data) > 0 and not vmax:
        vmax = numeric_data.max().max()
    # Get labels
    if label is True:
        labels = keys
    elif isinstance(label, str):
        labels = [label]
    elif isinstance(label, Sequence):
        labels = label
    else:
        raise ValueError("Invalid label value. Must be a bool, str, or a sequence of strings.")
    # Compute coordinates for annotations
    start_lat = attrs["offset"] + attrs["depth"] * gap
    end_lat = start_lat + attrs["depth"] * width * data.shape[1]
    lats = np.linspace(start_lat, end_lat, data.shape[1] + 1)
    lons = np.linspace(0, 2 * np.pi, data.shape[0] + 1)
    lons = lons - np.pi / len(attrs["leaves"])
    # Covert to RGB array
    rgb_array = []
    if is_array:
        if data.shape[0] == data.shape[1]:
            data = data.loc[attrs["leaves"], reversed(attrs["leaves"])]
            end_lat = start_lat + attrs["depth"] + 2 * np.pi
            lats = np.linspace(start_lat, end_lat, data.shape[1] + 1)
        for col in data.columns:
            rgb_array.append(_series_to_rgb_array(data[col], cmap, vmin=vmin, vmax=vmax, na_color=na_color))
    else:
        for key in keys:
            if data[key].dtype == "category":
                colors = _get_categorical_colors(tdata, key, data[key], palette)
                rgb_array.append(_series_to_rgb_array(data[key], colors, na_color=na_color))
            else:
                rgb_array.append(_series_to_rgb_array(data[key], cmap, vmin=vmin, vmax=vmax, na_color=na_color))
    rgb_array = np.stack(rgb_array, axis=1)
    # Plot
    if attrs["polar"]:
        ax.pcolormesh(lons, lats, rgb_array.swapaxes(0, 1), zorder=2, **kwargs)
        ax.set_ylim(-0.2, end_lat)
    else:
        ax.pcolormesh(lats, lons, rgb_array, zorder=2, **kwargs)
        ax.set_xlim(-0.2, end_lat)
        labels_lats = np.linspace(start_lat, end_lat, len(labels) + 1)
        labels_lats = labels_lats + (end_lat - start_lat) / (len(labels) * 2)
        for idx, label in enumerate(labels):
            if is_array and len(labels) == 1:
                ax.text(labels_lats[idx], -0.1, label, ha="center", va="top")
                ax.set_ylim(-0.5, 2 * np.pi)
            else:
                ax.text(labels_lats[idx], -0.1, label, ha="center", va="top", rotation=90)
                ax.set_ylim(-1, 2 * np.pi)
    ax._attrs.update({"offset": end_lat})
    return ax


# For internal use
_annotation = annotation


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def tree(
    tdata: td.TreeData,
    key: str = None,
    nodes: str | Sequence[str] = None,
    annotation_keys: str | Sequence[str] = None,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    branch_color: str = "black",
    branch_linewidth: int | float | str = 1,
    node_color: str = "black",
    node_style: str = "o",
    node_size: int | float = 10,
    annotation_width: int | float = 0.05,
    cmap: str | mcolors.Colormap = "viridis",
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot a tree with branches, nodes, and annotations.

    Parameters
    ----------
    tdata
        The TreeData object.
    key
        The `obst` key of the tree to plot.
    nodes
        Either "all", "leaves", "internal", or a list of nodes to plot.
    annotation_keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to plot.
    polar
        Whether to plot the tree in polar coordinates.
    extend_branches
        Whether to extend branches so the tips are at the same depth.
    angled_branches
        Whether to plot branches at an angle.
    branch_color
        Either a color name, or a key for an attribute of the edges to color by.
    branch_linewidth
        Either an numeric width, or a key for an attribute of the edges to set the linewidth.
    node_color
        Either a color name, or a key for an attribute of the nodes to color by.
    node_style
        Either a marker name, or a key for an attribute of the nodes to set the marker.
    node_size
        Either an numeric size, or a key for an attribute of the nodes to set the size.
    annotation_width
        The width of the annotation bar relative to the tree.
    {common_plot_args}

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Plot branches
    ax = _branches(
        tdata,
        key=key,
        polar=polar,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        color=branch_color,
        linewidth=branch_linewidth,
        cmap=cmap,
        palette=palette,
        ax=ax,
    )
    # Plot nodes
    if nodes:
        ax = _nodes(
            tdata, nodes=nodes, color=node_color, style=node_style, size=node_size, cmap=cmap, palette=palette, ax=ax
        )
    # Plot annotations
    if annotation_keys:
        ax = _annotation(tdata, keys=annotation_keys, width=annotation_width, cmap=cmap, palette=palette, ax=ax)
    return ax
