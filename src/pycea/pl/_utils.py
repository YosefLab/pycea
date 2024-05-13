"""Plotting utilities"""

import collections.abc as cabc
import warnings

import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scanpy.plotting import palettes

from pycea._utils import get_root


def layout_tree(
    tree: nx.DiGraph,
    depth_key: str = "time",
    polar: bool = False,
    extend_branches: bool = True,
    angled_branches: bool = False,
):
    """Given a tree, computes the coordinates of the nodes and branches.

    Parameters
    ----------
    tree
        The `nx.DiGraph` representing the tree.
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
    # Get node depths
    n_leaves = 0
    root = get_root(tree)
    depths = {}
    for node in tree.nodes():
        if tree.out_degree(node) == 0:
            n_leaves += 1
        depths[node] = tree.nodes[node].get(depth_key)
    max_depth = max(depths.values())
    # Get node coordinates
    i = 0
    leaves = []
    node_coords = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        if tree.out_degree(node) == 0:
            lon = (i / n_leaves) * 2 * np.pi
            if extend_branches:
                node_coords[node] = (max_depth, lon)
            else:
                node_coords[node] = (depths[node], lon)
            leaves.append(node)
            i += 1
        else:
            children = list(tree.successors(node))
            min_lon = min(node_coords[child][1] for child in children)
            max_lon = max(node_coords[child][1] for child in children)
            node_coords[node] = (depths[node], (min_lon + max_lon) / 2)
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
    return node_coords, branch_coords, leaves, max_depth


def _get_default_categorical_colors(length):
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
                "The selected key has more than 103 categories. Uniform "
                "'grey' color will be used for all categories.",
                stacklevel=2,
            )
    colors_list = [mcolors.to_hex(palette[k], keep_alpha=True) for k in range(length)]
    return colors_list


def _get_categorical_colors(tdata, key, data, palette=None):
    """Get categorical colors for plotting."""
    # Ensure data is a category
    if not data.dtype.name == "category":
        data = data.astype("category")
    categories = data.cat.categories
    # Use default colors if no palette is provided
    if palette is None:
        colors_list = tdata.uns.get(key + "_colors", None)
        if colors_list is None or len(colors_list) > len(categories):
            colors_list = _get_default_categorical_colors(len(categories))
    # Use provided palette
    else:
        if isinstance(palette, str) and palette in plt.colormaps():
            # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
            cmap = plt.get_cmap(palette)
            colors_list = [mcolors.to_hex(x, keep_alpha=True) for x in cmap(np.linspace(0, 1, len(categories)))]
        elif isinstance(palette, cabc.Mapping):
            colors_list = [mcolors.to_hex(palette[k], keep_alpha=True) for k in categories]
        else:
            # check if palette is a list and convert it to a cycler, thus
            # it doesnt matter if the list is shorter than the categories length:
            if isinstance(palette, cabc.Sequence):
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
                        raise ValueError("The following color value of the given palette " f"is not valid: {color}")
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
    tdata.uns[key + "_colors"] = colors_list
    return dict(zip(categories, colors_list))
