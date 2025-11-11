from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import treedata as td
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from pycea.utils import _check_tree_overlap, get_keyed_edge_data, get_keyed_node_data, get_keyed_obs_data, get_trees

from ._docs import _doc_params, doc_common_plot_args
from ._legend import _categorical_legend, _cbar_legend, _render_legends
from ._utils import (
    _get_categorical_colors,
    _get_categorical_markers,
    _get_colors,
    _get_norm,
    _get_sizes,
    _series_to_rgb_array,
    layout_trees,
)


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def branches(
    tdata: td.TreeData,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    color: str = "black",
    linewidth: float | str = 0.5,
    depth_key: str = "depth",
    legend: bool | None = None,
    tree: str | Sequence[str] | None = None,
    cmap: str | mcolors.Colormap = "viridis",
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[Any, str] | None = None,
    vmax: float | str | None = None,
    vmin: float | str | None = None,
    na_color: str = "lightgrey",
    na_linewidth: float = 1,
    linewidths: Mapping[str, float] | tuple[float, float] = (0.1, 2),
    slot: Literal["obst", "obsp"] | None = None,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot the branches of a tree.

    Parameters
    ----------
    tdata
        The `treedata.TreeData` object.
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
    depth_key
        The key for the depth of the nodes.
    legend
        Whether to add a legend to the plot. By default, a legend is added if there
        are <= 20 distinct categories.
    {common_plot_args}
    na_color
        The color to use for edges with missing data.
    na_linewidth
        The linewidth to use for edges with missing data.
    slot
        Slot in TreeData object containing `color` and `linewidth` attributes.
        If `None`, searches `obst` when `alignment='leaves'`, searches `obst` and `obsp` otherwise.
    ax
        A matplotlib axes object. If `None`, a new figure and axes will be created.
    legend_kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`.
    kwargs
        Additional keyword arguments passed to :class:`matplotlib.collections.LineCollection`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    tdata._sanitize()
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"} if polar else None)
    elif (ax.name == "polar" and not polar) or (ax.name != "polar" and polar):
        warnings.warn("Polar setting of axes does not match requested type. Creating new axes.", stacklevel=2)
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"} if polar else None)
    ax = cast(Axes, ax)
    kwargs = kwargs if kwargs else {}
    trees = get_trees(tdata, tree_keys)
    # Get layout
    node_coords, branch_coords, leaves, depth = layout_trees(
        trees, depth_key=depth_key, polar=polar, extend_branches=extend_branches, angled_branches=angled_branches
    )
    segments = []
    edges = []
    legends = []
    max_categories = 0
    for edge, (lat, lon) in branch_coords.items():
        coords = np.array([lon, lat] if polar else [lat, lon]).T
        segments.append(coords)
        edges.append(edge)
    kwargs.update({"segments": segments})
    # Get colors
    if mcolors.is_color_like(color):
        kwargs.update({"color": color})
    elif isinstance(color, str):
        color_data = get_keyed_edge_data(tdata, color, tree_keys, slot=slot)[color]
        colors, color_legend, n_categories = _get_colors(
            tdata, color, color_data, edges, palette, cmap, vmin, vmax, na_color
        )
        kwargs.update({"color": colors})
        legends.append(color_legend)
        max_categories = max(max_categories, n_categories)
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the edges.")
    # Get linewidths
    if isinstance(linewidth, float | int):
        kwargs.update({"linewidth": linewidth})
    elif isinstance(linewidth, str):
        linewidth_data = get_keyed_edge_data(tdata, linewidth, tree_keys, slot=slot)[linewidth]
        scaled_widths, width_legend, n_categories = _get_sizes(
            tdata, linewidth, linewidth_data, edges, linewidths, na_value=na_linewidth
        )
        kwargs.update({"linewidth": scaled_widths})
        legends.append(width_legend)
        max_categories = max(max_categories, n_categories)
    else:
        raise ValueError("Invalid linewidth value. Must be int, float, or an str specifying an attribute of the edges.")
    # Plot
    ax.add_collection(LineCollection(zorder=1, **kwargs))
    if polar:
        ax.set_ylim(-depth * 0.05, depth * 1.05)
        ax.spines["polar"].set_visible(False)
    else:
        ax.set_ylim(-0.03 * np.pi, 2.03 * np.pi)
        ax.set_xlim(-depth * 0.05, depth * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax._attrs = {  # type: ignore
        "node_coords": node_coords,
        "leaves": leaves,
        "depth": depth,
        "offset": depth,
        "polar": polar,
        "tree_keys": list(trees.keys()),
    }
    # Add legends
    if legend is True or (legend is None and legends and max_categories <= 20):
        _render_legends(ax, legends, anchor_x=1.01, spacing=0.02, shared_kwargs=legend_kwargs)
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
    size: float | str = 10,
    legend: bool | None = None,
    tree: str | Sequence[str] | None = None,
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[Any, str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    vmax: float | str | None = None,
    vmin: float | str | None = None,
    markers: Sequence[str] | Mapping[str, str] | None = None,
    sizes: tuple[float, float] | Mapping[str, float] = (5, 50),
    na_color: str = "#FFFFFF00",
    na_style: str = "none",
    na_size: float = 0,
    slot: Literal["obst", "obs", "X"] | None = None,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
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
    legend
        Whether to add a legend to the plot. By default, a legend is added if there
        are <= 20 distinct categories.
    {common_plot_args}
    markers
        Object determining how to draw the markers for different levels of the style variable.
        You can pass a list of markers or a dictionary mapping levels of the style variable to markers.
    sizes
        Object determining how to draw the sizes for different levels of the size variable.
        You can pass a dictionary mapping levels to sizes, or a min, max tuple to use as a range.
    na_color
        The color to use for annotations with missing data.
    na_style
        The marker to use for annotations with missing data.
    na_size
        The size to use for annotations with missing data.
    ax
        A matplotlib axes object. If `None`, a new figure and axes will be created.
    slot
        Slot in TreeData object containing `color`, `style`, and `size` attributes.
        If `None`, searches `obst` when `alignment='leaves'`, searches `obs` and `X` otherwise.
    legend_kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`.
    kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    kwargs = kwargs if kwargs else {}
    if not ax:
        ax = plt.gca()
    ax = cast(Axes, ax)
    attrs = ax._attrs if hasattr(ax, "_attrs") else None  # type: ignore
    if not attrs:
        raise ValueError("Branches most be plotted with pycea.pl.branches before annotations can be plotted.")
    if not cmap:
        cmap = mpl.rcParams["image.cmap"]
    if not sizes:
        sizes = (5, 80)
    if tree is None:
        tree_keys = attrs["tree_keys"]
    else:
        tree_keys = tree
    if isinstance(tree_keys, str):
        tree_keys = [tree_keys]
    if not set(tree_keys).issubset(attrs["tree_keys"]):
        raise ValueError("Invalid tree key. Must be one of the keys used to plot the branches.")
    # Get nodes
    all_nodes = []
    for node in list(attrs["node_coords"].keys()):
        if node[0] in tree_keys:
            all_nodes.append(node)
    if nodes == "all":
        plot_nodes = all_nodes
    elif nodes == "leaves":
        plot_nodes = [node for node in all_nodes if node[1] in attrs["leaves"]]
    elif nodes == "internal":
        plot_nodes = [node for node in all_nodes if node[1] not in attrs["leaves"]]
    elif isinstance(nodes, Sequence):
        if isinstance(nodes, str):
            nodes = [nodes]
        if len(attrs["tree_keys"]) > 1 and len(tree_keys) > 1:
            raise ValueError("Multiple trees are present. To plot a list of nodes, you must specify the tree.")
        plot_nodes = [(tree_keys[0], node) for node in nodes]
        if set(plot_nodes).issubset(all_nodes):
            plot_nodes = list(plot_nodes)
        else:
            raise ValueError("Nodes must be a list of nodes in the tree.")
    else:
        raise ValueError("Invalid nodes value. Must be 'all', 'leaves', 'no_leaves', or a list of nodes.")
    # Get coordinates
    coords = np.vstack([attrs["node_coords"][node] for node in plot_nodes])
    if attrs["polar"]:
        kwargs.update({"x": coords[:, 1], "y": coords[:, 0]})
    else:
        kwargs.update({"x": coords[:, 0], "y": coords[:, 1]})
    kwargs_list = []
    legends = []
    max_categories = 0
    # Get colors
    if mcolors.is_color_like(color):
        kwargs.update({"color": color})
    elif isinstance(color, str):
        color_data = get_keyed_node_data(tdata, color, tree_keys, slot=slot)[color]
        if len(color_data) == 0:
            raise ValueError(f"Key {color!r} is not present in any node.")
        colors, color_legend, n_categories = _get_colors(
            tdata, color, color_data, plot_nodes, palette, cmap, vmin, vmax, na_color, marker_type="marker"
        )
        kwargs.update({"color": colors})
        legends.append(color_legend)
        max_categories = max(max_categories, n_categories)
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the nodes.")
    # Get sizes
    if isinstance(size, float | int):
        kwargs.update({"s": size})
    elif isinstance(size, str):
        size_data = get_keyed_node_data(tdata, size, tree_keys, slot=slot)[size]
        if len(size_data) == 0:
            raise ValueError(f"Key {size!r} is not present in any node.")
        marker_sizes, size_legend, n_categories = _get_sizes(
            tdata, size, size_data, plot_nodes, sizes, na_value=na_size, marker_type="marker"
        )
        kwargs.update({"s": marker_sizes})
        legends.append(size_legend)
        max_categories = max(max_categories, n_categories)
    else:
        raise ValueError("Invalid size value. Must be int, float, or an str specifying an attribute of the nodes.")
    # Get markers
    if style in mmarkers.MarkerStyle.markers:
        kwargs.update({"marker": style})
    elif isinstance(style, str):
        style_data = get_keyed_node_data(tdata, style, tree_keys)[style]
        if len(style_data) == 0:
            raise ValueError(f"Key {style!r} is not present in any node.")
        marker_map = _get_categorical_markers(tdata, style, style_data, markers)
        styles = [marker_map[style_data[node]] if node in style_data.index else na_style for node in plot_nodes]
        for marker_style in set(styles):
            style_kwargs = {}
            idx = [i for i, x in enumerate(styles) if x == marker_style]
            for key, value in kwargs.items():
                if isinstance(value, list | np.ndarray):
                    style_kwargs[key] = [value[i] for i in idx]
                else:
                    style_kwargs[key] = value
            style_kwargs.update({"marker": marker_style})
            kwargs_list.append(style_kwargs)
        legends.append(_categorical_legend(style, marker_map=marker_map, type="marker"))
    else:
        raise ValueError("Invalid style value. Must be a marker name, or an str specifying an attribute of the nodes.")
    # Plot
    if len(kwargs_list) > 0:
        for kwargs in kwargs_list:
            ax.scatter(**kwargs)
    else:
        ax.scatter(**kwargs)
    # Add legends
    if legend is True or (legend is None and max_categories <= 20):
        _render_legends(ax, legends, anchor_x=1.01, spacing=0.02, shared_kwargs=legend_kwargs)
    return ax


# For internal use
_nodes = nodes


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def annotation(
    tdata: td.TreeData,
    keys: str | Sequence[str] | None = None,
    width: float = 0.05,
    gap: float = 0.03,
    label: bool | str | Sequence[str] | None = True,
    layer: str | None = None,
    border_width: float = 0,
    legend: bool | None = None,
    tree: str | Sequence[str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[Any, str] | None = None,
    vmax: float | str | None = None,
    vmin: float | str | None = None,
    share_cmap: bool = False,
    na_color: str = "white",
    slot: Literal["obs", "obsm", "obsp", "X"] | None = None,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> Axes:
    """\
    Plot leaf annotations for a tree.

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
    layer
        Name of the TreeData object layer to use. If `None`, `tdata.X` is plotted.
    border_width
        The width of the border around the annotation bar.
    legend
        Whether to add a legend to the plot. By default, a legend is added if there
        are <= 20 distinct categories.
    {common_plot_args}
    share_cmap
        If `True`, all numeric keys will share the same colormap.
    na_color
        The color to use for annotations with missing data.
    slot
        Slot in TreeData object containing `keys`. If `None`, searches `obs`, `X`, `obsm`, and `obsp` in that order.
    ax
        A matplotlib axes object. If `None`, a new figure and axes will be created.
    legend_kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`.
    kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    if tree:  # TODO: Annotate only the leaves for the given tree
        pass
    if not ax:
        ax = plt.gca()
    ax = cast(Axes, ax)
    attrs = ax._attrs if hasattr(ax, "_attrs") else None  # type: ignore
    if not attrs:
        raise ValueError("Branches most be plotted with pycea.pl.branches before annotations can be plotted.")
    if not keys:
        raise ValueError("No keys provided. Please provide one or more keys to plot.")
    keys = [keys] if isinstance(keys, str) else keys
    if not cmap:
        color_map = mpl.rcParams["image.cmap"]
    color_map = plt.get_cmap(cmap)
    leaves = attrs["leaves"]
    # Get data
    data, is_array, is_square = get_keyed_obs_data(tdata, keys, layer=layer, slot=slot)
    data = data.reindex(leaves, axis=0)
    if is_square:
        data = data.reindex(leaves, axis=1)
    numeric_data = data.select_dtypes(exclude="category")
    if vmin is not None and vmax is not None:
        share_cmap = True
    if share_cmap and len(numeric_data.columns) > 1:
        if vmin is None:
            vmin = numeric_data.min().min()
        if vmax is None:
            vmax = numeric_data.max().max()
    norm = None
    # Get labels
    if label is True:
        labels = keys
    elif label is False:
        labels = []
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
    lons = np.linspace(0, 2 * np.pi, len(leaves) + 1)
    lons = lons - np.pi / len(leaves)
    # Covert to RGB array
    rgb_array = []
    legends = []
    max_categories = 0
    if is_array:  # single cmap for all columns
        label = labels[0] if labels is not None else keys[0]
        if is_square:
            data = data.loc[leaves, list(reversed(leaves))]
            end_lat = start_lat + attrs["depth"] * 2 * np.pi * width / 0.05
            lats = np.linspace(start_lat, end_lat, data.shape[1] + 1)
        if data.loc[:, data.columns[0]].dtype == "category":  # all columns are categorical with same categories
            max_categories = len(data.loc[:, data.columns[0]].cat.categories)
            color_map = _get_categorical_colors(
                tdata, keys[0], data.loc[leaves, data.columns[0]], palette, subset=False
            )
            legends.append(_categorical_legend(label, color_map, type="patch"))
        else:
            norm = _get_norm(vmin, vmax, data=data)
            share_cmap = True
        for col in data.columns:
            rgb_array.append(_series_to_rgb_array(data.loc[leaves, col], color_map, norm=norm, na_color=na_color))
    else:  # separate cmaps for each key
        for i, key in enumerate(keys):
            label = key
            if labels is not None:
                label = labels[i] if i < len(labels) else key
            if data[key].dtype == "category":
                max_categories = max(max_categories, len(data[key].cat.categories))
                key_color_map = _get_categorical_colors(tdata, key, data.loc[leaves, key], palette)
                rgb_array.append(_series_to_rgb_array(data.loc[leaves, key], key_color_map, na_color=na_color))
                legends.append(_categorical_legend(label, key_color_map, type="patch"))
            else:
                norm = _get_norm(vmin, vmax, data=data.loc[leaves, key])
                rgb_array.append(_series_to_rgb_array(data.loc[leaves, key], color_map, norm=norm, na_color=na_color))
                if not share_cmap:
                    legends.append(_cbar_legend(label, color_map, norm))
    # Add shared cmap
    if share_cmap and norm is not None:
        if labels is not None:
            label = labels[0]
        elif is_array:
            label = keys[0]
        elif layer is not None:
            label = layer
        elif keys[0] in tdata.var_names:
            label = "expression"
        else:
            label = "values"
        legends.append(_cbar_legend(label, color_map, norm))
    # Plot
    rgb_array = np.stack(rgb_array, axis=1)
    if attrs["polar"]:
        ax.pcolormesh(lons, lats, rgb_array.swapaxes(0, 1), zorder=2, **kwargs)
        ax.set_ylim(-attrs["depth"] * 0.05, end_lat)
    else:
        ax.pcolormesh(lats, lons, rgb_array, zorder=2, **kwargs)
        ax.set_xlim(-attrs["depth"] * 0.05, end_lat + attrs["depth"] * width * 0.5)
        # Add border
        ax.plot(
            [lats[0], lats[0], lats[-1], lats[-1], lats[0]],
            [lons[0], lons[-1], lons[-1], lons[0], lons[0]],
            color="black",
            linewidth=border_width,
        )
        # Add labels
        if labels and len(labels) > 0:
            labels_lats = np.linspace(start_lat, end_lat, len(labels) + 1)
            labels_lats = labels_lats + (end_lat - start_lat) / (len(labels) * 2)
            existing_ticks = ax.get_xticks()
            existing_labels = [label.get_text() for label in ax.get_xticklabels()]
            ax.set_xticks(np.append(existing_ticks, labels_lats[:-1]))
            ax.set_xticklabels(existing_labels + list(labels))
            for xlabel in ax.get_xticklabels()[len(existing_ticks) :]:
                if is_array and len(labels) == 1:
                    xlabel.set_rotation(0)
                else:
                    xlabel.set_rotation(90)
    # Add legends
    if legend is True or (legend is None and max_categories <= 20):
        _render_legends(ax, legends, anchor_x=1.01, spacing=0.02, shared_kwargs=legend_kwargs)
    ax._attrs.update({"offset": end_lat})  # type: ignore
    return ax


# For internal use
_annotation = annotation


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def tree(
    tdata: td.TreeData,
    keys: str | Sequence[str] | None = None,
    nodes: str | Sequence[str] | None = None,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    depth_key: str = "depth",
    branch_color: str = "black",
    branch_linewidth: float | str = 0.5,
    node_color: str = "black",
    node_style: str = "o",
    node_size: str | float = 10,
    annotation_width: float = 0.05,
    tree: str | Sequence[str] | None = None,
    legend: bool | Mapping[str, bool] | None = None,
    cmap: str | mcolors.Colormap = "viridis",
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[Any, str] | None = None,
    vmax: float | str | None = None,
    vmin: float | str | None = None,
    share_cmap: bool = False,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """\
    Plot a tree with branches, nodes, and annotations.

    Parameters
    ----------
    tdata
        The TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` annotations.
    nodes
        Either "all", "leaves", "internal", or a list of nodes to plot. Defaults to "internal" if node color, style, or size is set.
    polar
        Whether to plot the tree in polar coordinates.
    extend_branches
        Whether to extend branches so the tips are at the same depth.
    angled_branches
        Whether to plot branches at an angle.
    depth_key
        The key for the depth of the nodes.
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
    legend
        Whether to show a legend. By default, a legend is added if there
        are <= 20 distinct categories. Can also be a dictionary with keys "branch", "node", and "annotation"
        to control the legend for each component separately.
    annotation_width
        The width of the annotation bar relative to the tree.
    {common_plot_args}
    share_cmap
        If `True`, all numeric keys will share the same colormap.
    ax
        A matplotlib axes object. If `None`, a new figure and axes will be created.
    legend_kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`.

    Returns
    -------
    ax - The axes that the plot was drawn on.
    """  # noqa: D205
    # Setup
    branch_legend = legend.get("branch", None) if isinstance(legend, Mapping) else legend
    node_legend = legend.get("node", None) if isinstance(legend, Mapping) else legend
    annotation_legend = legend.get("annotation", None) if isinstance(legend, Mapping) else legend
    # Plot branches
    ax = _branches(
        tdata,
        polar=polar,
        depth_key=depth_key,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        color=branch_color,
        linewidth=branch_linewidth,
        tree=tree,
        cmap=cmap,
        palette=palette,
        vmax=vmax,
        vmin=vmin,
        legend=branch_legend,
        legend_kwargs=legend_kwargs,
        ax=ax,
    )
    # Plot nodes
    if nodes is None and (node_color != "black" or node_style != "o" or node_size != 10):
        nodes = "internal"
    if nodes:
        ax = _nodes(
            tdata,
            nodes=nodes,
            color=node_color,
            style=node_style,
            size=node_size,
            tree=tree,
            cmap=cmap,
            palette=palette,
            vmax=vmax,
            vmin=vmin,
            legend=node_legend,
            legend_kwargs=legend_kwargs,
            ax=ax,
        )
    # Plot annotations
    if keys:
        ax = _annotation(
            tdata,
            keys=keys,
            legend=annotation_legend,
            width=annotation_width,
            cmap=cmap,
            palette=palette,
            vmax=vmax,
            vmin=vmin,
            share_cmap=share_cmap,
            ax=ax,
            legend_kwargs=legend_kwargs,
        )
    return ax
