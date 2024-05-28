from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import treedata as td
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from pycea.utils import get_keyed_edge_data, get_keyed_node_data, get_keyed_obs_data, get_trees

from ._docs import _doc_params, doc_common_plot_args
from ._utils import (
    _get_categorical_colors,
    _get_categorical_markers,
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
    linewidth: int | float | str = 1,
    depth_key: str = "depth",
    tree: str | Sequence[str] | None = None,
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
    tree
        The `obst` key or keys of the trees to plot. If `None`, all trees are plotted.
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
    # Setup
    tree_keys = tree
    if not ax:
        ax = plt.gca()
    if (ax.name == "polar" and not polar) or (ax.name != "polar" and polar):
        warnings.warn("Polar setting of axes does not match requested type. Creating new axes.", stacklevel=2)
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"} if polar else None)
    kwargs = kwargs if kwargs else {}
    trees = get_trees(tdata, tree_keys)
    # Get layout
    node_coords, branch_coords, leaves, depth = layout_trees(
        trees, depth_key=depth_key, polar=polar, extend_branches=extend_branches, angled_branches=angled_branches
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
        color_data = get_keyed_edge_data(tdata, color, tree_keys)[color]
        print(color_data)
        if len(color_data) == 0:
            raise ValueError(f"Key {color!r} is not present in any edge.")
        color_data.index = color_data.index.map(lambda x: f"{x[0]}-{x[1][0]}-{x[1][1]}")
        if color_data.dtype.kind in ["i", "f"]:
            norm = plt.Normalize(vmin=color_data.min(), vmax=color_data.max())
            cmap = plt.get_cmap(cmap)
            colors = [cmap(norm(color_data[edge])) if edge in color_data.index else na_color for edge in edges]
            kwargs.update({"color": colors})
        else:
            if color in tdata.obs.columns:
                if tdata.obs[color].dtype.kind not in ["i", "f"]:
                    tdata.obs[color] = tdata.obs[color].astype("category")
                    if set(color_data.unique()).issubset(tdata.obs[color].cat.categories):
                        color_data = pd.Series(
                            pd.Categorical(color_data, categories=tdata.obs[color].cat.categories),
                            index=color_data.index,
                        )
            cmap = _get_categorical_colors(tdata, color, color_data, palette)
            colors = [cmap[color_data[edge]] if edge in color_data.index else na_color for edge in edges]
            kwargs.update({"color": colors})
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the edges.")
    # Get linewidths
    if isinstance(linewidth, (int, float)):
        kwargs.update({"linewidth": linewidth})
    elif isinstance(linewidth, str):
        linewidth_data = get_keyed_edge_data(tdata, linewidth, tree_keys)[linewidth]
        if len(linewidth_data) == 0:
            raise ValueError(f"Key {linewidth!r} is not present in any edge.")
        linewidth_data.index = linewidth_data.index.map(lambda x: f"{x[0]}-{x[1][0]}-{x[1][1]}")
        if linewidth_data.dtype.kind in ["i", "f"]:
            linewidths = [linewidth_data[edge] if edge in linewidth_data.index else na_linewidth for edge in edges]
            kwargs.update({"linewidth": linewidths})
        else:
            raise ValueError("Invalid linewidth data type. Edge attribute must be int or float")
    else:
        raise ValueError("Invalid linewidth value. Must be int, float, or an str specifying an attribute of the edges.")
    # Plot
    ax.add_collection(LineCollection(zorder=1, **kwargs))
    if polar:
        ax.set_ylim((-depth * 0.05, depth * 1.05))
        ax.spines["polar"].set_visible(False)
    else:
        ax.set_ylim((-0.03 * np.pi, 2.03 * np.pi))
        ax.set_xlim((-depth * 0.05, depth * 1.05))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax._attrs = {
        "node_coords": node_coords,
        "leaves": leaves,
        "depth": depth,
        "offset": depth,
        "polar": polar,
        "tree_keys": list(trees.keys()),
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
    tree: str | Sequence[str] | None = None,
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
    tree
        The `obst` key or keys of the trees to plot. If `None`, all trees are plotted.
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
    if tree is None:
        tree_keys = attrs["tree_keys"]
    else:
        tree_keys = tree
    if isinstance(tree_keys, str):
        tree_keys = [tree_keys]
    if not set(tree_keys).issubset(attrs["tree_keys"]):
        raise ValueError("Invalid tree key. Must be one of the keys used to plot the branches.")
    trees = get_trees(tdata, attrs["tree_keys"])
    # Get nodes
    all_nodes = set()
    for node in list(attrs["node_coords"].keys()):
        if any(node.startswith(key) for key in tree_keys):
            all_nodes.add(node)
    leaves = set(attrs["leaves"])
    if nodes == "all":
        nodes = list(all_nodes)
    elif nodes == "leaves":
        nodes = list(all_nodes.intersection(leaves))
    elif nodes == "internal":
        nodes = list(all_nodes.difference(leaves))
    elif isinstance(nodes, Sequence):
        if len(attrs["tree_keys"]) > 1 and len(tree_keys) > 1:
            raise ValueError("Multiple trees are present. To plot a list of nodes, you must specify the tree.")
        nodes = [f"{tree_keys[0]}-{node}" for node in nodes]
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
        color_data = get_keyed_node_data(tdata, color, tree_keys)[color]
        if len(color_data) == 0:
            raise ValueError(f"Key {color!r} is not present in any node.")
        color_data.index = color_data.index.map("-".join)
        if color_data.dtype.kind in ["i", "f"]:
            if not vmin:
                vmin = color_data.min()
            if not vmax:
                vmax = color_data.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colors = [cmap(norm(color_data[node])) if node in color_data.index else na_color for node in nodes]
            kwargs.update({"color": colors})
        else:
            if color in tdata.obs.columns:
                if tdata.obs[color].dtype.kind not in ["i", "f"]:
                    tdata.obs[color] = tdata.obs[color].astype("category")
                    if set(color_data.unique()).issubset(tdata.obs[color].cat.categories):
                        color_data = pd.Series(
                            pd.Categorical(color_data, categories=tdata.obs[color].cat.categories),
                            index=color_data.index,
                        )
            cmap = _get_categorical_colors(tdata, color, color_data, palette)
            colors = [cmap[color_data[node]] if node in color_data.index else na_color for node in nodes]
            kwargs.update({"color": colors})
    else:
        raise ValueError("Invalid color value. Must be a color name, or an str specifying an attribute of the nodes.")
    # Get sizes
    if isinstance(size, (int, float)):
        kwargs.update({"s": size})
    elif isinstance(size, str):
        size_data = get_keyed_node_data(tdata, size, tree_keys)[size]
        if len(size_data) == 0:
            raise ValueError(f"Key {size!r} is not present in any node.")
        size_data.index = size_data.index.map("-".join)
        sizes = [size_data[node] if node in size_data.index else na_size for node in nodes]
        kwargs.update({"s": sizes})
    else:
        raise ValueError("Invalid size value. Must be int, float, or an str specifying an attribute of the nodes.")
    # Get markers
    if style in mmarkers.MarkerStyle.markers:
        kwargs.update({"marker": style})
    elif isinstance(style, str):
        style_data = get_keyed_node_data(tdata, style, tree_keys)[style]
        if len(style_data) == 0:
            raise ValueError(f"Key {style!r} is not present in any node.")
        style_data.index = style_data.index.map("-".join)
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
    border_width: int | float = 0,
    tree: str | Sequence[str] | None = None,
    cmap: str | mcolors.Colormap = None,
    palette: cycler.Cycler | mcolors.ListedColormap | Sequence[str] | Mapping[str] | None = None,
    vmax: int | float | None = None,
    vmin: int | float | None = None,
    na_color: str = "white",
    ax: Axes | None = None,
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
    border_width
        The width of the border around the annotation bar.
    tree
        The `obst` key or keys of the trees to plot. If `None`, all trees are plotted.
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
    if tree:  # TODO: Annotate only the leaves for the given tree
        pass
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
    leaves = attrs["leaves"]
    # Get data
    data, is_array = get_keyed_obs_data(tdata, keys)
    data = data.loc[leaves]
    numeric_data = data.select_dtypes(exclude="category")
    if len(numeric_data) > 0 and not vmin:
        vmin = numeric_data.min().min()
    if len(numeric_data) > 0 and not vmax:
        vmax = numeric_data.max().max()
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
    if is_array:
        if data.shape[0] == data.shape[1]:
            data = data.loc[leaves, reversed(leaves)]
            end_lat = start_lat + attrs["depth"] + 2 * np.pi
            lats = np.linspace(start_lat, end_lat, data.shape[1] + 1)
        for col in data.columns:
            rgb_array.append(_series_to_rgb_array(data.loc[col], cmap, vmin=vmin, vmax=vmax, na_color=na_color))
    else:
        for key in keys:
            if data[key].dtype == "category":
                colors = _get_categorical_colors(tdata, key, data.loc[leaves, key], palette)
                rgb_array.append(_series_to_rgb_array(data.loc[leaves, key], colors, na_color=na_color))
            else:
                rgb_array.append(
                    _series_to_rgb_array(data.loc[leaves, key], cmap, vmin=vmin, vmax=vmax, na_color=na_color)
                )
    rgb_array = np.stack(rgb_array, axis=1)
    # Plot
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
            ax.set_xticklabels(existing_labels + labels)
            for label in ax.get_xticklabels()[len(existing_ticks) :]:
                if is_array and len(labels) == 1:
                    label.set_rotation(0)
                else:
                    label.set_rotation(90)
    ax._attrs.update({"offset": end_lat})
    return ax


# For internal use
_annotation = annotation


@_doc_params(
    common_plot_args=doc_common_plot_args,
)
def tree(
    tdata: td.TreeData,
    keys: str | Sequence[str] = None,
    tree: str | Sequence[str] | None = None,
    nodes: str | Sequence[str] = None,
    polar: bool = False,
    extend_branches: bool = False,
    angled_branches: bool = False,
    depth_key: str = "depth",
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
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` annotations.
    tree
        The `obst` key or keys of the trees to plot. If `None`, all trees are plotted.
    nodes
        Either "all", "leaves", "internal", or a list of nodes to plot.
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
        polar=polar,
        depth_key=depth_key,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        color=branch_color,
        linewidth=branch_linewidth,
        tree=tree,
        cmap=cmap,
        palette=palette,
        ax=ax,
    )
    # Plot nodes
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
            ax=ax,
        )
    # Plot annotations
    if keys:
        ax = _annotation(tdata, keys=keys, width=annotation_width, cmap=cmap, palette=palette, ax=ax)
    return ax
