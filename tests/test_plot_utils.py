import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.pl._utils import (
    _get_categorical_colors,
    _get_categorical_markers,
    _get_default_categorical_colors,
    _series_to_rgb_array,
    layout_tree,
)


# Test layout_tree
def test_layout_empty_tree():
    tree = nx.DiGraph()
    with pytest.raises(ValueError):
        layout_tree(tree)


def test_layout_tree():
    tree = nx.DiGraph()
    tree.add_nodes_from(
        [("A", {"time": 0}), ("B", {"time": 1}), ("C", {"time": 2}), ("D", {"time": 2}), ("E", {"time": 2})]
    )
    edges = [("A", "B"), ("B", "C"), ("B", "D"), ("A", "E")]
    tree.add_edges_from(edges)
    node_coords, branch_coords, leaves, max_depth = layout_tree(tree, extend_branches=True)
    assert sorted(leaves) == ["C", "D", "E"]
    assert max_depth == 2
    assert set(branch_coords.keys()) == set(edges)
    assert branch_coords[("B", "C")][0] == [1, 1, 2]
    assert branch_coords[("B", "C")][1] == [node_coords["B"][1], node_coords["C"][1], node_coords["C"][1]]


def test_layout_polar_coordinates():
    tree = nx.DiGraph()
    tree.add_nodes_from(
        [
            ("A", {"time": 0}),
            ("B", {"time": 1}),
            ("C", {"time": 2}),
            ("D", {"time": 2}),
        ]
    )
    tree.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])
    node_coords, branch_coords, _, _ = layout_tree(tree, polar=True)
    assert len(branch_coords[("B", "C")][1]) > 2
    assert np.mean(branch_coords[("B", "C")][0][:-2]) == 1


def test_layout_angled_branches():
    tree = nx.DiGraph()
    tree.add_nodes_from([("A", {"time": 0}), ("B", {"time": 1})])
    tree.add_edge("A", "B")
    _, branch_coords, _, _ = layout_tree(tree, angled_branches=True)
    assert len(branch_coords[("A", "B")][1]) == 2


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
    rgb_array = _series_to_rgb_array(numeric_data, colors, vmin=0, vmax=2)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)
    # Test with missing data
    numeric_data = pd.Series([0, np.nan, 2])
    rgb_array = _series_to_rgb_array(numeric_data, colors, vmin=0, vmax=2)
    expected = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1]])
    np.testing.assert_almost_equal(rgb_array, expected, decimal=2)
