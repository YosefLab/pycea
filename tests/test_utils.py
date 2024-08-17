import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.utils import (
    check_tree_has_key,
    get_keyed_edge_data,
    get_keyed_leaf_data,
    get_keyed_node_data,
    get_keyed_obs_data,
    get_keyed_obsm_data,
    get_leaves,
    get_root,
    get_subtree_leaves,
)


@pytest.fixture
def tree():
    t = nx.DiGraph()
    t.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    nx.set_node_attributes(t, {"A": 1, "B": 2, "C": 2, "D": 4, "E": 4}, "value")
    nx.set_node_attributes(t, {"A": "red", "B": "red", "D": "blue", "E": "blue"}, "color")
    nx.set_edge_attributes(t, {("A", "B"): 5, ("B", "D"): 3, ("C", "E"): 4}, "weight")
    nx.set_edge_attributes(t, {("A", "B"): "red", ("B", "D"): "red", ("C", "E"): "blue"}, "color")
    yield t


@pytest.fixture
def tdata(tree):
    tdata = td.TreeData(
        obs=pd.DataFrame({"value": ["1", "2"]}, index=["D", "E"]),
        obst={"tree": tree},
        obsm={"spatial": pd.DataFrame([[0, 0], [1, 1]], index=["D", "E"])},
    )
    yield tdata


def test_get_root(tree):
    # Test with an empty graph
    assert get_root(nx.DiGraph()) is None
    # Test with a non-empty graph
    assert get_root(tree) == "A"
    # Test with a single node
    single_node_tree = nx.DiGraph()
    single_node_tree.add_node("A")
    assert get_root(single_node_tree) == "A"


def test_get_leaves(tree):
    assert get_leaves(tree) == ["D", "E"]
    # test with empty graph
    assert get_leaves(nx.DiGraph()) == []


def test_get_subtree_leaves(tree):
    assert get_subtree_leaves(tree, "B") == ["D"]
    assert get_subtree_leaves(tree, "A") == ["D", "E"]
    # Test with a single node
    single_node_tree = nx.DiGraph()
    single_node_tree.add_node("A")
    assert get_subtree_leaves(single_node_tree, "A") == ["A"]


def test_get_keyed_edge_data(tdata):
    data = get_keyed_edge_data(tdata, ["weight", "color"])
    assert data.columns.tolist() == ["weight", "color"]
    assert data.index.names == ["tree", "edge"]
    assert data["weight"].to_list() == [5, 3, 4]


def test_get_keyed_node_data(tdata):
    data = get_keyed_node_data(tdata, ["value", "color"])
    assert data.columns.tolist() == ["value", "color"]
    assert data.index.names == ["tree", "node"]
    assert data["value"].to_list() == [1, 2, 2, 4, 4]


def test_get_keyed_leaf_data(tdata):
    data = get_keyed_leaf_data(tdata, ["value", "color"])
    print(data)
    assert data.columns.tolist() == ["value", "color"]
    assert data["value"].tolist() == [4, 4]
    assert data["color"].tolist() == ["blue", "blue"]


def test_get_keyed_obs_data_valid_keys(tdata):
    data, is_array = get_keyed_obs_data(tdata, "value")
    assert not is_array
    assert data["value"].tolist() == ["1", "2"]
    # Automatically converts object columns to category
    assert data["value"].dtype == "category"
    assert tdata.obs["value"].dtype == "category"


def test_get_keyed_obs_data_array(tdata):
    data, is_array = get_keyed_obs_data(tdata, ["spatial"])
    assert data.columns.tolist() == [0, 1]
    assert data[0].tolist() == [0, 1]
    assert is_array
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == 2


def test_get_keyed_obs_data_invalid_keys(tdata):
    with pytest.raises(ValueError):
        get_keyed_obs_data(tdata, ["bad"])
    with pytest.raises(ValueError):
        get_keyed_obs_data(tdata, ["value", "spatial"])


def test_check_tree_has_key(tree):
    check_tree_has_key(tree, "value")
    with pytest.raises(ValueError):
        check_tree_has_key(tree, "bad")


def test_get_keyed_obsm_data(tdata):
    data = get_keyed_obsm_data(tdata, "spatial")
    assert data.columns.tolist() == [0, 1]
    assert data.index.tolist() == ["D", "E"]
    assert data[0].tolist() == [0, 1]
    assert data[1].tolist() == [0, 1]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
