import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.pl._utils import (
    ScaledNormalize,
    _get_categorical_colors,
    _get_categorical_markers,
    _get_colors,
    _get_default_categorical_colors,
    _get_norm,
    _get_sizes,
    _series_to_rgb_array,
    layout_trees,
)


# Test layout_tree
def test_layout_empty_tree():
    tree = nx.DiGraph()
    with pytest.raises(ValueError):
        layout_trees({"tree": tree})


def test_layout_tree():
    tree = nx.DiGraph()
    tree.add_nodes_from(
        [("A", {"depth": 0}), ("B", {"depth": 1}), ("C", {"depth": 2}), ("D", {"depth": 2}), ("E", {"depth": 2})]
    )
    edges = [("A", "B"), ("B", "C"), ("B", "D"), ("A", "E")]
    expected_edges = [(("tree"), ("A", "B")), (("tree"), ("B", "C")), (("tree"), ("B", "D")), (("tree"), ("A", "E"))]
    tree.add_edges_from(edges)
    node_coords, branch_coords, leaves, max_depth = layout_trees({"tree": tree}, extend_branches=True)
    assert sorted(leaves) == ["C", "D", "E"]
    assert max_depth == 2
    assert set(branch_coords.keys()) == set(expected_edges)
    assert branch_coords[("tree", ("B", "C"))][0] == [1, 1, 2]
    assert branch_coords[("tree", ("B", "C"))][1] == [
        node_coords[("tree", "B")][1],
        node_coords[("tree", "C")][1],
        node_coords[("tree", "C")][1],
    ]


def test_layout_multiple_trees():
    tree1 = nx.DiGraph([("root", "A")])
    tree1.nodes["root"]["depth"] = 0
    tree1.nodes["A"]["depth"] = 1
    tree2 = nx.DiGraph([("root", "B")])
    tree2.nodes["root"]["depth"] = 0
    tree2.nodes["B"]["depth"] = 2
    expected_edges = [(1, ("root", "A")), (2, ("root", "B"))]
    node_coords, branch_coords, leaves, max_depth = layout_trees(
        {1: tree1, 2: tree2, "empty": nx.DiGraph()}, extend_branches=False
    )
    assert leaves == ["A", "B"]
    assert max_depth == 2
    assert set(branch_coords.keys()) == set(expected_edges)
    assert branch_coords[(1, ("root", "A"))][0] == [0, 0, 1]


def test_layout_polar_coordinates():
    tree = nx.DiGraph()
    tree.add_nodes_from(
        [
            ("A", {"depth": 0}),
            ("B", {"depth": 1}),
            ("C", {"depth": 2}),
            ("D", {"depth": 2}),
        ]
    )
    tree.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])
    node_coords, branch_coords, _, _ = layout_trees({"tree": tree}, polar=True)
    assert len(branch_coords[("tree", ("B", "C"))][1]) > 2
    assert np.mean(branch_coords[("tree", ("B", "C"))][0][:-2]) == 1


def test_layout_angled_branches():
    tree = nx.DiGraph()
    tree.add_nodes_from([("A", {"time": 0}), ("B", {"time": 1})])
    tree.add_edge("A", "B")
    _, branch_coords, _, _ = layout_trees({"tree": tree}, angled_branches=True, depth_key="time")
    assert len(branch_coords[("tree", ("A", "B"))][1]) == 2


# Test _get_default_categorical_colors
def test_default_palettes():
    # Small
    colors = _get_default_categorical_colors(5)
    assert colors[0] == "#1f77b4ff"
    colors = _get_default_categorical_colors(25)
    assert colors[0] == "#023fa5ff"
    colors = _get_default_categorical_colors(50)
    assert colors[0] == "#ffff00ff"


def test_overflow_palette():
    # Test requesting more colors than the largest palette
    with pytest.warns(Warning, match="more than 103 categories"):
        colors = _get_default_categorical_colors(104)
    assert len(colors) == 104
    assert all(color == mcolors.to_hex("grey", keep_alpha=True) for color in colors)


# Test _get_categorical_colors
@pytest.fixture
def empty_tdata():
    yield td.TreeData()


@pytest.fixture
def category_data():
    yield pd.Series(["apple", "banana", "cherry"])


def test_palette_types(empty_tdata, category_data):
    # String
    colors = _get_categorical_colors(empty_tdata, "fruit", category_data, "tab10")
    assert colors["apple"] == "#1f77b4ff"
    # Dict
    palette = {"apple": "red", "banana": "yellow", "cherry": "pink"}
    colors = _get_categorical_colors(empty_tdata, "fruit", category_data, palette)
    assert colors["apple"] == "#ff0000ff"
    # List not enough
    palette = ["red", "yellow"]
    with pytest.warns(Warning, match="palette colors is smaller"):
        colors = _get_categorical_colors(empty_tdata, "fruit", category_data, palette)
        assert colors["apple"] == "#ff0000ff"


def test_invalid_palette(empty_tdata, category_data):
    with pytest.warns(Warning, match="palette colors is smaller"):
        with pytest.raises(ValueError):
            _get_categorical_colors(empty_tdata, "fruit", category_data, ["bad"])


def test_pallete_in_uns(empty_tdata, category_data):
    palette_hex = {"apple": "#ff0000ff", "banana": "#ffff00ff", "cherry": "#ff69b4ff"}
    colors = _get_categorical_colors(empty_tdata, "fruit", category_data, palette_hex)
    assert "fruit_colors" in empty_tdata.uns
    assert empty_tdata.uns["fruit_colors"] == list(palette_hex.values())
    colors = _get_categorical_colors(empty_tdata, "fruit", category_data)
    assert colors == palette_hex


# Test _get_categorical_markers
def test_markers_types(empty_tdata, category_data):
    # None
    markers = _get_categorical_markers(empty_tdata, "fruit", category_data)
    assert markers["apple"] == "o"
    # Dict
    marker_dict = {"apple": "s", "banana": "o", "cherry": "o"}
    colors = _get_categorical_markers(empty_tdata, "fruit", category_data, marker_dict)
    assert colors["apple"] == "s"
    # List not enough
    marker_list = ["s", "o"]
    with pytest.warns(Warning, match="Length of markers"):
        markers = _get_categorical_markers(empty_tdata, "fruit", category_data, marker_list)
    assert markers["apple"] == "s"


def test_markers_in_uns(empty_tdata, category_data):
    marker_dict = {"apple": "s", "banana": "o", "cherry": "o"}
    markers = _get_categorical_markers(empty_tdata, "fruit", category_data, marker_dict)
    assert "fruit_markers" in empty_tdata.uns
    assert empty_tdata.uns["fruit_markers"] == list(marker_dict.values())
    markers = _get_categorical_markers(empty_tdata, "fruit", category_data)
    assert markers == marker_dict


# Test _series_to_rgb_array
def test_series_to_rgb_discrete(category_data):
    colors = {"apple": "#ff0000ff", "banana": "#ffff00ff", "cherry": "#ff69b4ff"}
    rgb_array = _series_to_rgb_array(category_data, colors)
    expected = np.array([[1, 0, 0], [1, 1, 0], [1, 0.41176471, 0.70588235]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)
    # Test with missing data
    category_data = pd.Series(["apple", pd.NA])
    rgb_array = _series_to_rgb_array(category_data, colors)
    expected = np.array([[1, 0, 0], [0.5, 0.5, 0.5]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)


def test_series_to_rgb_numeric():
    numeric_data = pd.Series([0, 1, 2])
    colors = mcolors.ListedColormap(["red", "yellow", "blue"])
    norm = mcolors.Normalize(vmin=0, vmax=2)
    rgb_array = _series_to_rgb_array(numeric_data, colors, norm=norm)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)
    # Test with missing data
    numeric_data = pd.Series([0, np.nan, 2])
    rgb_array = _series_to_rgb_array(numeric_data, colors, norm=norm)
    expected = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)


def test_get_norm_defaults_and_errors():
    # Test Series input
    s = pd.Series([1, 3, 5])
    norm_s = _get_norm(data=s)
    assert norm_s.vmin == 1
    assert norm_s.vmax == 5
    # Test percentile input
    norm_p = _get_norm(data=s, vmin="p10", vmax="p90")
    assert norm_p.vmin == 1.4
    assert norm_p.vmax == 4.6
    # Test DataFrame input
    df = pd.DataFrame({"a": [0, 2], "b": [4, 6]})
    norm_df = _get_norm(data=df)
    assert norm_df.vmin == 0
    assert norm_df.vmax == 6
    # Missing parameters
    with pytest.raises(ValueError):
        _get_norm()


def test_scaled_normalize_and_get_bins():
    # Test scaling functionality
    sn = ScaledNormalize(vmin=0, vmax=10, scale=(2, 8))
    arr = sn([0, 5, 10])
    expected = np.array([2, 5, 8])
    np.testing.assert_allclose(arr, expected)
    # Test bin generation
    bins = sn.get_bins(num_bins=3)
    assert isinstance(bins, dict)
    # Number of bins may vary slightly; ensure at least requested number
    assert len(bins) >= 3
    # Ensure scale_min and scale_max are included among bin values
    values = list(bins.values())
    assert sn.scale_min in values
    assert sn.scale_max in values


def test_get_colors_numeric():
    tdata = td.TreeData()
    data = pd.Series([0, 1, 2], index=["a", "b", "c"])
    indices = ["a", "b", "c", "d"]
    colors, legend, ncat = _get_colors(tdata, "num", data, indices, cmap="viridis")
    assert isinstance(colors, list)
    assert len(colors) == 4
    assert colors[-1] == "lightgrey"
    assert ncat == 0
    assert isinstance(legend, dict)


def test_get_colors_categorical():
    tdata = td.TreeData()
    # Prepare obs for categorical reference
    tdata.obs["cat"] = pd.Categorical(["x", "y", "z"], categories=["x", "y", "z"])
    data = pd.Series(["x", "y", "z"], index=[0, 1, 2])
    indices = [0, 1, 2, 3]
    palette = {"x": "red", "y": "green", "z": "blue"}
    colors, legend, ncat = _get_colors(tdata, "cat", data, indices, palette=palette)
    expected = [mcolors.to_hex(c, keep_alpha=True) for c in ["red", "green", "blue"]]
    assert colors[:3] == expected
    assert colors[-1] == "lightgrey"
    assert ncat == 3
    assert isinstance(legend, dict)


def test_get_sizes_numeric_and_categorical():
    tdata = td.TreeData()
    # Numeric mapping
    num_data = pd.Series([1, 3, 5], index=[0, 1, 2])
    indices = [0, 2, 3]
    sizes_num, legend_num, ncat_num = _get_sizes(tdata, "sz", num_data, indices, mapping=(10, 20))
    assert isinstance(sizes_num, list)
    assert len(sizes_num) == 3
    assert isinstance(sizes_num[0], float)
    assert sizes_num[-1] == 6  # default na_value
    assert ncat_num == 0
    assert isinstance(legend_num, dict)
    # Categorical mapping
    cat_data = pd.Series(["low", "high", "med"], index=[0, 1, 2])
    size_map = {"low": 5, "high": 10, "med": 15}
    sizes_cat, legend_cat, ncat_cat = _get_sizes(tdata, "sz", cat_data, indices, mapping=size_map)
    assert sizes_cat[:2] == [5, 15]
    assert sizes_cat[-1] == 6
    # Categories include na_value as a size category
    assert ncat_cat == 3
    assert isinstance(legend_cat, dict)
