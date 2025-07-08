"""Plotting utilities"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scanpy.plotting import palettes

from pycea.utils import check_tree_has_key, get_leaves, get_root


def layout_nodes_and_branches(
    tree: nx.DiGraph,
    leaf_coords: dict[str, tuple[float, float]],
    depth_key: str = "depth",
    polar: bool = False,
    angled_branches: bool = False,
) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], tuple[list[float], list[float]]]]:
    """Given a tree and leaf coordinates, computes the coordinates of the nodes and branches.

    Parameters
    ----------
    tree
        The `nx.DiGraph` representing the tree.
    leaf_coords
        A dictionary mapping leaves to their coordinates.
    depth_key
        The node attribute to use as the depth of the nodes.
    polar
        Whether to plot the tree in polar coordinates.
    extend_branches
        Whether to extend branches so the tips are at the same depth.
    angled_branches
        Whether to plot branches at an angle.

    Returns
    -------
    node_coords
        A dictionary mapping nodes to their coordinates.
    branch_coords
        A dictionary mapping edges to their coordinates.
    """
    # Get node coordinates
    node_coords = copy.copy(leaf_coords)
    for node in nx.dfs_postorder_nodes(tree, get_root(tree)):
        if tree.out_degree(node) != 0:
            children = list(tree.successors(node))
            min_lon = min(node_coords[child][1] for child in children)
            max_lon = max(node_coords[child][1] for child in children)
            node_coords[node] = (tree.nodes[node].get(depth_key), (min_lon + max_lon) / 2)
    # Get branch coordinates
    branch_coords = {}
    for parent, child in tree.edges():
        parent_coord, child_coord = node_coords[parent], node_coords[child]
        if angled_branches:
            branch_coords[(parent, child)] = ([parent_coord[0], child_coord[0]], [parent_coord[1], child_coord[1]])
        else:
            branch_coords[(parent, child)] = (
                [parent_coord[0], parent_coord[0], child_coord[0]],
                [parent_coord[1], child_coord[1], child_coord[1]],
            )
    # Interpolate branch coordinates
    min_angle = np.pi / 50
    if polar:
        for parent, child in branch_coords:
            lats, lons = branch_coords[(parent, child)]
            angle = abs(lons[0] - lons[1])
            if angle > min_angle:
                # interpolate points
                inter_lons = np.linspace(lons[0], lons[1], int(np.ceil(angle / min_angle)))
                inter_lats = [lats[0]] * len(inter_lons)
                branch_coords[(parent, child)] = (np.append(inter_lats, lats[-1]), np.append(inter_lons, lons[-1]))
    return node_coords, branch_coords


def layout_trees(
    trees: Mapping[str, nx.DiGraph],
    depth_key: str = "depth",
    polar: bool = False,
    extend_branches: bool = True,
    angled_branches: bool = False,
) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    dict[tuple[str, str], tuple[list[float], list[float]]],
    list[str],
    float,
]:
    """Given a list of trees, computes the coordinates of the nodes and branches.

    Parameters
    ----------
    trees
        A dictionary mapping tree names to `nx.DiGraph` representing the trees.
    depth_key
        The node attribute to use as the depth of the nodes.
    polar
        Whether to plot the tree in polar coordinates.
    extend_branches
        Whether to extend branches so the tips are at the same depth.
    angled_branches
        Whether to plot branches at an angle.

    Returns
    -------
    node_coords
        A dictionary mapping nodes to their coordinates.
    branch_coords
        A dictionary mapping edges to their coordinates.
    leaves
        A list of the leaves of the tree.
    max_depth
        The maximum depth of the tree.
    """
    # Get leaf coordinates
    leaves = []
    depths = []
    for _, tree in trees.items():
        check_tree_has_key(tree, depth_key)
        tree_leaves = get_leaves(tree)
        leaves.extend(tree_leaves)
        depths.extend(tree.nodes[leaf].get(depth_key) for leaf in tree_leaves)
        if len(depths) != len(leaves):
            raise ValueError(f"Every node in the tree must have a {depth_key} attribute. ")
    max_depth = max(depths)
    n_leaves = len(leaves)
    leaf_coords = {}
    for i in range(n_leaves):
        lon = (i / n_leaves) * 2 * np.pi
        if extend_branches:
            leaf_coords[leaves[i]] = (max_depth, lon)
        else:
            leaf_coords[leaves[i]] = (depths[i], lon)
    # Layout trees
    node_coords = {}
    branch_coords = {}
    for key, tree in trees.items():
        tree_node_coords, tree_branch_coords = layout_nodes_and_branches(
            tree, leaf_coords, depth_key, polar, angled_branches
        )
        node_coords.update({(key, node): coords for node, coords in tree_node_coords.items()})
        branch_coords.update({(key, edge): coords for edge, coords in tree_branch_coords.items()})
    return node_coords, branch_coords, leaves, max_depth


def _get_default_categorical_colors(length: int) -> list[str]:
    """Get default categorical colors for plotting."""
    # check if default matplotlib palette has enough colors
    if len(mpl.rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = mpl.rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]
    # if not, use scanpy default palettes
    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            warnings.warn(
                "The selected key has more than 103 categories. Uniform 'grey' color will be used for all categories.",
                stacklevel=2,
            )
    colors_list = [mcolors.to_hex(palette[k], keep_alpha=True) for k in range(length)]
    return colors_list


def _get_categorical_colors(
    tdata: td.TreeData, key: str, data: Any, palette: Any | None = None, save: bool = True
) -> dict[Any, Any]:
    """Get categorical colors for plotting."""
    # Check type of data
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    # Ensure data is a category
    if not data.dtype.name == "category":
        data = data.astype("category")
    categories = data.cat.categories
    # Use default colors if no palette is provided
    if palette is None:
        colors_list = tdata.uns.get(key + "_colors", None)
        if (colors_list is None) or (len(colors_list) < len(categories)):
            colors_list = _get_default_categorical_colors(len(categories))
    # Use provided palette
    else:
        if isinstance(palette, str) and palette in mpl.colormaps:
            # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
            cmap = mpl.colormaps.get_cmap(palette)
            colors_list = [mcolors.to_hex(x, keep_alpha=True) for x in cmap(np.linspace(0, 1, len(categories)))]
        elif isinstance(palette, Mapping):
            colors_list = [mcolors.to_hex(palette[k], keep_alpha=True) for k in categories]
        else:
            # check if palette is a list and convert it to a cycler, thus
            # it doesnt matter if the list is shorter than the categories length:
            if isinstance(palette, Sequence):
                if len(palette) < len(categories):
                    warnings.warn(
                        "Length of palette colors is smaller than the number of "
                        f"categories (palette length: {len(palette)}, "
                        f"categories length: {len(categories)}. "
                        "Some categories will have the same color.",
                        stacklevel=2,
                    )
                # check that colors are valid
                _color_list = []
                for color in palette:
                    if not mcolors.is_color_like(color):
                        raise ValueError(f"The following color value of the given palette is not valid: {color}")
                    _color_list.append(color)
                palette = cycler.cycler(color=_color_list)
            if not isinstance(palette, cycler.Cycler):
                raise ValueError(
                    "Please check that the value of 'palette' is a valid "
                    "matplotlib colormap string (eg. Set2), a  list of color names "
                    "or a cycler with a 'color' key."
                )
            if "color" not in palette.keys:
                raise ValueError("Please set the palette key 'color'.")
            cc = palette()
            colors_list = [mcolors.to_hex(next(cc)["color"], keep_alpha=True) for x in range(len(categories))]
    # store colors in tdata
    if save and len(categories) <= len(palettes.default_102):
        tdata.uns[key + "_colors"] = colors_list
    return dict(zip(categories, colors_list, strict=False))


def _get_categorical_markers(
    tdata: td.TreeData, key: str, data: pd.Series, markers: Mapping | Sequence | None = None
) -> dict[Any, Any]:
    """Get categorical markers for plotting."""
    default_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "P", "*", "h", "H", "X"]
    # Ensure data is a category
    if not data.dtype.name == "category":
        data = data.astype("category")
    categories = data.cat.categories
    # Use default markers if no markers are provided
    if markers is None:
        markers_list = tdata.uns.get(key + "_markers", None)
        if markers_list is None or len(markers_list) > len(categories):
            markers_list = default_markers[: len(categories)]
    # Use provided markers
    else:
        if isinstance(markers, Mapping):
            markers_list = [markers[k] for k in categories]
        else:
            if not isinstance(markers, Sequence):
                raise ValueError("Please check that the value of 'markers' is a valid list of marker names.")
            if len(markers) < len(categories):
                warnings.warn(
                    "Length of markers is smaller than the number of "
                    f"categories (markers length: {len(markers)}, "
                    f"categories length: {len(categories)}. "
                    "Some categories will have the same marker.",
                    stacklevel=2,
                )
                markers_list = list(markers) * (len(categories) // len(markers) + 1)
            else:
                markers_list = markers[: len(categories)]
    # store markers in tdata
    tdata.uns[key + "_markers"] = markers_list
    return dict(zip(categories, markers_list, strict=False))


def _series_to_rgb_array(
    series: Any,
    colors: dict[Any, Any] | mcolors.Colormap,
    vmin: float | None = None,
    vmax: float | None = None,
    na_color: str = "#808080",
) -> np.ndarray:
    """Converts a pandas Series to an N x 3 numpy array based using a color map."""
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    if isinstance(colors, dict):
        # Map using the dictionary
        color_series = series.map(colors).astype("object")
        color_series[series.isna()] = na_color
        rgb_array = np.array([mcolors.to_rgb(color) for color in color_series])
    elif isinstance(colors, mcolors.ListedColormap | mcolors.LinearSegmentedColormap):
        # Normalize and map values if cmap is a ListedColormap
        if vmin is not None and vmax is not None:
            norm = mcolors.Normalize(vmin, vmax)
            colors.set_bad(na_color)
            color_series = colors(norm(series))
            rgb_array = np.array(color_series)[:, :3]
        else:
            raise ValueError("vmin and vmax must be specified when using a ListedColormap.")
    else:
        raise ValueError("cmap must be either a dictionary or a ListedColormap.")
    return rgb_array


def _check_tree_overlap(
    tdata: td.TreeData,
    tree_keys: str | Sequence[str] | None = None,
) -> None:
    """Check single tree is requested when allow_overlap is True"""
    if tree_keys is None:
        if tdata.allow_overlap and len(tdata.obst.keys()) > 1:
            raise ValueError("Must specify a tree when tdata.allow_overlap is True.")
    elif isinstance(tree_keys, str):
        pass
    elif isinstance(tree_keys, Sequence):
        if tdata.allow_overlap:
            raise ValueError("Cannot request multiple trees when tdata.allow_overlap is True.")
    else:
        raise ValueError("Tree keys must be a string, list of strings, or None.")
