import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy as sp
import treedata as td

from pycea.tl.tree_distance import tree_distance


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D")])
    nx.set_node_attributes(tree1, {"root": 0, "A": 3, "B": 1, "C": 2, "D": 3}, "depth")
    tree2 = nx.DiGraph([("root", "E"), ("root", "F")])
    nx.set_node_attributes(tree2, {"root": 0, "E": 1, "F": 1}, "depth")
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["A", "C", "D", "E", "F"]),
        obst={"tree1": tree1, "tree2": tree2, "empty": nx.DiGraph()},
        obsp={"connectivities": sp.sparse.csr_matrix(([1, 1, 1], ([0, 0, 3], [0, 4, 4])), shape=(5, 5))},
    )
    yield tdata


def test_sparse_tree_distance(tdata):
    dist = tree_distance(tdata, "depth", metric="path", copy=True)
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert dist.shape == (5, 5)
    assert dist[0, 1] == 5
    assert dist[0, 2] == 6
    assert "tree_distances" in tdata.uns.keys()
    assert tdata.uns["tree_distances"]["params"]["metric"] == "path"
    tree_distance(tdata, "depth", metric="lca", key_added="lca")
    assert isinstance(tdata.obsp["lca_distances"], sp.sparse.csr_matrix)
    assert isinstance(tdata.obsp["lca_connectivities"], sp.sparse.csr_matrix)
    assert tdata.obsp["lca_distances"].shape == (5, 5)
    assert tdata.obsp["lca_distances"][0, 1] == 0
    assert tdata.obsp["lca_distances"][0, 2] == 0
    assert tdata.obsp["lca_distances"][1, 2] == 1


def test_pairwise_tree_distance(tdata):
    tdata_subset = tdata[tdata.obs.tree == "tree1"].copy()
    dist = tree_distance(tdata_subset, "depth", metric="path", copy=True)
    expected = np.array([[0, 5, 6], [5, 0, 3], [6, 3, 0]])
    assert isinstance(dist, np.ndarray)
    np.testing.assert_array_equal(dist, expected)


def test_obs_tree_distance(tdata):
    tree_distance(tdata, "depth", obs="A", metric="path")
    assert tdata.obs.loc["A", "tree_distances"] == 0
    assert tdata.obs.loc["C", "tree_distances"] == 5
    assert pd.isna(tdata.obs.loc["E", "tree_distances"])


def test_select_obs_tree_distance(tdata):
    tree_distance(tdata, "depth", obs=["A", "C"], metric="path")
    assert isinstance(tdata.obsp["tree_distances"], sp.sparse.csr_matrix)
    assert len(tdata.obsp["tree_distances"].data) == 4
    assert tdata.obsp["tree_distances"][0, 1] == 5
    assert tdata.obsp["tree_distances"][0, 0] == 0
    dist = tree_distance(tdata, "depth", obs=[("A", "C")], metric="path", copy=True, update=False)
    assert len(tdata.obsp["tree_distances"].data) == 1
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert dist[0, 1] == 5


def test_sampled_tree_distance(tdata):
    tree_distance(tdata, "depth", sample_n=3, random_state=0, metric="path")
    assert isinstance(tdata.obsp["tree_distances"], sp.sparse.csr_matrix)
    assert len(tdata.obsp["tree_distances"].data) == 3
    assert tdata.obsp["tree_distances"].shape == (5, 5)
    assert tdata.obsp["tree_distances"].data.tolist() == [0, 3, 2]
    assert tdata.obsp["tree_connectivities"].shape == (5, 5)
    assert len(tdata.obsp["tree_connectivities"].data) == 3


def test_connected_tree_distance(tdata):
    tree_distance(tdata, "depth", connect_key="connectivities", metric="path")
    assert isinstance(tdata.obsp["tree_distances"], sp.sparse.csr_matrix)
    assert tdata.obsp["tree_distances"].shape == (5, 5)
    assert len(tdata.obsp["tree_distances"].data) == 2
    assert tdata.obsp["tree_connectivities"].sum() == 2
    assert tdata.obsp["tree_distances"].data.tolist() == [0, 2]


def test_update_tree_distance(tdata):
    tree_distance(tdata, "depth", sample_n=2, random_state=0, metric="path")
    assert tdata.obsp["tree_distances"].data.tolist() == [3, 2]
    tree_distance(tdata, "depth", sample_n=1, random_state=1, metric="path", update=True)
    assert tdata.obsp["tree_distances"].data.tolist() == [5, 3, 2]
    tree_distance(tdata, "depth", sample_n=2, random_state=0, metric="path", update=False)
    assert tdata.obsp["tree_distances"].data.tolist() == [3, 2]
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", sample_n=2, metric="lca", update=True)


def test_tree_distance_invalid(tdata):
    with pytest.raises(ValueError):
        tree_distance(tdata, "bad", metric="path")
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", obs=1, metric="path")
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", obs=[1], metric="path")
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", obs=[("A",)], metric="path")
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", obs=[("A", "B", "C")], metric="path")
    with pytest.raises(ValueError):
        tree_distance(tdata, "depth", metric="bad")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
