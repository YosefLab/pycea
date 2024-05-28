import networkx as nx
import pandas as pd
import pytest

from pycea.utils import get_keyed_edge_data, get_keyed_node_data, get_keyed_obs_data, get_leaves, get_root


@pytest.fixture
def tree():
    t = nx.DiGraph()
    t.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    nx.set_node_attributes(t, {"A": 1, "B": 2, "C": None, "D": 4, "E": 5}, "value")
    nx.set_edge_attributes(t, {("A", "B"): 5, ("A", "C"): None, ("B", "D"): 3, ("C", "E"): 4}, "weight")
    yield t


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


def test_get_keyed_edge_data(tdata):
    data = get_keyed_edge_data(tdata, ["length", "clade"])
    assert data.columns.tolist() == ["length", "clade"]


def test_get_keyed_node_data(tdata):
    data = get_keyed_node_data(tdata, ["x", "y", "clade"])
    assert data.columns.tolist() == ["x", "y", "clade"]


def test_get_keyed_obs_data_valid_keys(tdata):
    data, is_array = get_keyed_obs_data(tdata, ["clade", "x", "0"])
    assert not is_array
    assert data.columns.tolist() == ["clade", "x", "0"]
    # Automatically converts object columns to category
    assert data["clade"].dtype == "category"
    assert tdata.obs["clade"].dtype == "category"


def test_get_keyed_obs_data_array(tdata):
    data, is_array = get_keyed_obs_data(tdata, ["spatial"])
    assert is_array
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == 2
    data, is_array = get_keyed_obs_data(tdata, ["spatial_distance"])
    assert data.shape == (tdata.n_obs, tdata.n_obs)


def test_get_keyed_obs_data_invalid_keys(tdata):
    with pytest.raises(ValueError):
        get_keyed_obs_data(tdata, ["clade", "x", "0", "invalid_key"])
    with pytest.raises(ValueError):
        get_keyed_obs_data(tdata, ["clade", "spatial_distance"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
