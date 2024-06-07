import networkx as nx
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
        obs=pd.DataFrame(index=["A", "C", "D", "E", "F"]), obst={"tree1": tree1, "tree2": tree2, "empty": nx.DiGraph()}
    )
    yield tdata


def test_tree_distance(tdata):
    dist = tree_distance(tdata, "depth", metric="path", copy=True)
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert dist.shape == (5, 5)
    assert dist[0, 1] == 5
    assert dist[0, 2] == 6
    tree_distance(tdata, "depth", metric="lca", key_added="lca_depth")
    assert isinstance(tdata.obsp["lca_depth"], sp.sparse.csr_matrix)
    assert tdata.obsp["lca_depth"].shape == (5, 5)
    assert tdata.obsp["lca_depth"][0, 1] == 0
    assert tdata.obsp["lca_depth"][0, 2] == 0
    assert tdata.obsp["lca_depth"][1, 2] == 1


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
    dist = tree_distance(tdata, "depth", obs=[("A", "C")], metric="path", copy=True)
    assert len(tdata.obsp["tree_distances"].data) == 1
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert dist[0, 1] == 5


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
