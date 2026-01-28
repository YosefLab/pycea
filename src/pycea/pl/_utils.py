"""Plotting utilities"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scanpy.plotting import palettes

from pycea.utils import check_tree_has_key, get_leaves, get_root

from ._legend import _categorical_legend, _cbar_legend


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
            node_coords[node] = (tree.nodes[node].get(depth_key), (min_lon + max_lon) / 2)  # type: ignore
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
            if (angle > min_angle) & (not angled_branches):
                # interpolate points
                inter_lons = np.linspace(lons[0], lons[1], int(np.ceil(angle / min_angle)))
                inter_lats = [lats[0]] * len(inter_lons)
                branch_coords[(parent, child)] = (np.append(inter_lats, lats[-1]), np.append(inter_lons, lons[-1]))
    return node_coords, branch_coords


def layout_trees(
    trees: Mapping[Any, nx.DiGraph],
    depth_key: str = "depth",
    polar: bool = False,
    extend_branches: bool = True,
    angled_branches: bool = False,
) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    dict[tuple[Any, tuple[str, str]], tuple[list[float], list[float]]],
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
    tdata: td.TreeData, key: str, data: Any, palette: Any | None = None, save: bool = True, subset: bool = True
) -> dict[Any, Any]:
    """Get categorical colors for plotting.

    Parameters
    ----------
    tdata
        The TreeData object containing the data.
    key
        The key in `tdata.obs` to use for the categorical variable.
    data
        The data to use for the categorical variable.
    palette
        The palette to use for the categorical variable. If `None`, the default palette is used.
    save
        Whether to save the colors in `tdata.uns[key + "_colors"]`.
    subset
        Whether to subset the data to the categories present in data.
    """
    # Check type of data
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    # Ensure data is a category
    if not data.dtype.name == "category":
        data = data.astype("category")
    categories = data.cat.categories
    # Use default colors if no palette is provided
    if palette is None:
        colors_list = tdata.uns.get(f"{key}_colors", None)
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
    colormap = dict(zip(categories, colors_list, strict=False))
    if subset:
        colormap = {k: v for k, v in colormap.items() if k in data.unique()}
    return colormap


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


def _get_norm(
    vmin: float | str | None = None,
    vmax: float | str | None = None,
    data: Any | None = None,
) -> mcolors.Normalize:
    """Get a normalization object for color mapping."""
    if isinstance(vmin, float | int):
        vmin_ = vmin
    elif vmin is None and data is not None:
        vmin_ = data.min().min() if isinstance(data, pd.DataFrame) else data.min()
    elif isinstance(vmin, str) and data is not None and vmin.startswith("p"):
        vmin_ = float(np.percentile(data, float(vmin[1:])))
    else:
        raise ValueError("vmin must be specified or data must be provided.")
    if isinstance(vmax, float | int):
        vmax_ = vmax
    elif vmax is None and data is not None:
        vmax_ = data.max().max() if isinstance(data, pd.DataFrame) else data.max()
    elif isinstance(vmax, str) and data is not None and vmax.startswith("p"):
        vmax_ = float(np.percentile(data, float(vmax[1:])))
    else:
        raise ValueError("vmax must be specified or data must be provided.")
    return mcolors.Normalize(vmin=vmin_, vmax=vmax_)


def _series_to_rgb_array(
    series: Any,
    colors: dict[Any, Any] | mcolors.Colormap,
    norm: mcolors.Normalize | None = None,
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
        if norm is None:
            raise ValueError("Normalization must be provided when using a ListedColormap.")
        colors.set_bad(na_color)
        color_series = colors(norm(series))
        rgb_array = np.array(color_series)[:, :3]
    else:
        raise ValueError("cmap must be either a dictionary or a ListedColormap.")
    return rgb_array


def _get_colors(
    tdata: td.TreeData,
    key: str,
    data: pd.Series,
    indicies: Sequence[Any],
    palette: Any | None = None,
    cmap: str | mcolors.Colormap | None = None,
    vmin: float | str | None = None,
    vmax: float | str | None = None,
    na_color: str = "lightgrey",
    marker_type: str = "line",
) -> tuple[list[Any], dict[str, Any], int]:
    """Get colors for plotting."""
    if len(data) == 0:
        raise ValueError(f"Key {key!r} is not present in any edge.")
    if data.dtype.kind in ["i", "f"]:  # Numeric
        norm = _get_norm(vmin=vmin, vmax=vmax, data=data)
        color_map = plt.get_cmap(cmap)
        colors = [color_map(norm(data[i])) if i in data.index else na_color for i in indicies]
        legend = _cbar_legend(key, color_map, norm)
        n_categories = 0
    else:  # Categorical
        if key in tdata.obs.columns and isinstance(tdata.obs[key].dtype, pd.CategoricalDtype):
            categories = tdata.obs[key].cat.categories
            if set(data.unique()).issubset(categories):
                data = pd.Series(
                    pd.Categorical(data, categories=categories, ordered=True),
                    index=data.index,
                )
        color_map = _get_categorical_colors(tdata, str(key), data, palette)
        colors = [color_map[data[i]] if (i in data.index and pd.notna(data.at[i])) else na_color for i in indicies]
        legend = _categorical_legend(key, {k: v for k, v in color_map.items() if v in colors}, type=marker_type)
        n_categories = len(data.unique())
    return colors, legend, n_categories


def _get_sizes(
    tdata: td.TreeData,
    key: str,
    data: pd.Series,
    indicies: Sequence[Any],
    mapping: tuple[float, float] | Mapping[str, float] = (5, 80),
    na_value: float = 6,
    marker_type: str = "line",
) -> tuple[list[float], dict[str, Any], int]:
    """Get sizes for plotting."""
    if len(data) == 0:
        raise ValueError(f"Key {key!r} is not present in any edge.")
    if isinstance(mapping, tuple) and data.dtype.kind in ["i", "f"]:  # Numeric
        size_norm = ScaledNormalize(vmin=data.min(), vmax=data.max(), scale=mapping)
        sizes = [size_norm(data.loc[i]) if (i in data.index and pd.notna(data.at[i])) else na_value for i in indicies]
        legend = _categorical_legend(key, size_map=size_norm.get_bins(), type=marker_type)
        n_categories = 0
    elif isinstance(mapping, Mapping) and data.dtype.kind not in ["i", "f"]:  # Categorical
        sizes = [mapping[data.loc[i]] if (i in data.index and pd.notna(data.at[i])) else na_value for i in indicies]
        n_categories = len(set(sizes))
        legend = _categorical_legend(key, size_map={k: v for k, v in mapping.items() if v in sizes}, type=marker_type)
    else:
        raise ValueError(
            "Invalid `size` parameter values. Must be (min, max) tuple for numeric data, or a mapping for categorical data."
        )
    return sizes, legend, n_categories


class ScaledNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, scale=(0.0, 1.0), clip=False):
        """Scaled normalization.

        Parameters
        ----------
        vmin
            The minimum value for normalization.
        vmax
            The maximum value for normalization.
        scale
            A tuple defining the scale range to map normalized values to.
        clip
            Whether to clip values outside the range [vmin, vmax].
        """
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.scale_min = scale[0]
        self.scale_max = scale[1]

    def __call__(self, value, clip=None):
        """Normalize the value and scale it to the specified range."""
        result = super().__call__(value, clip=clip)
        return self.scale_min + result * (self.scale_max - self.scale_min)

    def get_bins(self, num_bins=6):
        """Get pretty bin edges (as strings) mapped to normalized values."""
        if self.vmin is None or self.vmax is None:
            raise ValueError("vmin and vmax must be set to get bins.")
        # 1) build an extended range
        span = self.vmax - self.vmin
        pad = 0.05 * span
        lo = max(self.vmin - pad, 0)
        hi = self.vmax + pad
        # 2) let MaxNLocator pick “nice” tick positions
        locator = mticker.MaxNLocator(
            nbins=num_bins,
            steps=[1, 2, 2.5, 5, 10],
            prune=None,
            integer=False,
            min_n_ticks=1,
        )
        raw_bins = np.array(locator.tick_values(lo, hi))
        # 3) decide if they’re effectively all integers
        if np.allclose(raw_bins, raw_bins.astype(int), atol=1e-12):
            pretty = raw_bins.astype(int)
            labels = [str(int(b)) for b in pretty]
        else:
            # find the smallest positive step to set decimal precision
            diffs = np.diff(raw_bins)
            pos = diffs[diffs > 0]
            if pos.size:
                min_step = pos.min()
                precision = max(0, -int(math.floor(math.log10(min_step))) + 1)
            else:
                precision = 6

            pretty = np.round(raw_bins, precision)
            labels = []
            for b in pretty:
                s = f"{b:.{precision}f}".rstrip("0").rstrip(".")
                labels.append(s)
        # 4) map the string labels to the normalized values
        return {label: self(b) for label, b in zip(labels, pretty, strict=False)}
