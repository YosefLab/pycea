import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.tl.tree_neighbors import tree_neighbors


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "F")])
    nx.set_node_attributes(tree1, {"root": 0, "A": 2, "B": 1, "C": 3, "D": 3, "E": 3, "F": 3}, "depth")
    tree2 = nx.DiGraph([("root", "G"), ("root", "H"), ("G", "I"), ("G", "J")])
    nx.set_node_attributes(tree2, {"root": 0, "G": 1, "H": 1, "I": 3, "J": 2}, "depth")
    nx.set_node_attributes(tree2, {"root": 3, "G": 2, "H": 2, "I": 0, "J": 1}, "time")
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["C", "D", "E", "F", "I", "J", "H"]),
        obst={"tree1": tree1, "tree2": tree2, "empty": nx.DiGraph()},
    )
    yield tdata


def test_tree_neighbors_max(tdata):
    result = tree_neighbors(tdata, max_dist=3, metric="path", copy=True)
    assert isinstance(result, tuple)
    if isinstance(result, tuple):
        dist, _ = result
    assert tdata.obsp["tree_connectivities"].sum() == 10
    assert np.sum(dist > 0) == 10
    assert "tree_neighbors" in tdata.uns.keys()
    assert tdata.uns["tree_neighbors"]["params"]["metric"] == "path"
    tree_neighbors(tdata, max_dist=2, metric="path")
    assert tdata.obsp["tree_connectivities"].sum() == 6
    tree_neighbors(tdata, max_dist=2, metric="lca", tree="tree2", key_added="lca", depth_key="time")
    assert tdata.obsp["lca_connectivities"].sum() == 2


def test_tree_neighbors_n(tdata):
    tree_neighbors(tdata, n_neighbors=2, metric="path")
    assert tdata.obsp["tree_connectivities"].sum() == 14
    tree_neighbors(tdata, n_neighbors=3, metric="path")
    assert tdata.obsp["tree_connectivities"].sum() == 18
    tree_neighbors(tdata, n_neighbors=2, metric="lca", tree="tree2", key_added="lca", depth_key="time")
    assert tdata.obsp["lca_connectivities"].sum() == 6


def test_select_tree_neighbors(tdata):
    tree_neighbors(tdata, n_neighbors=2, metric="path", obs="C")
    assert tdata.obs.query("tree_neighbors").index.tolist() == ["C"]
    tree_neighbors(tdata, n_neighbors=3, metric="path", obs=["C", "D"], random_state=0)
    assert tdata.obsp["tree_connectivities"].sum() == 2


def test_update_tree_neighbors(tdata):
    tree_neighbors(tdata, n_neighbors=3, metric="path")
    tree_neighbors(tdata, n_neighbors=2, metric="path", update=True)
    assert tdata.obsp["tree_connectivities"].sum() == 14  # connectivities are updated
    assert (tdata.obsp["tree_distances"] > 0).sum() == 18  # but distances are not
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=2, metric="lca", tree="tree2", update=True)
    tree_neighbors(tdata, n_neighbors=2, metric="lca", tree="tree2", update=False)
    assert tdata.obsp["tree_connectivities"].sum() == 6


def test_tree_neighbors_distances(tdata):
    c_idx = tdata.obs_names.get_loc("C")
    d_idx = tdata.obs_names.get_loc("D")
    f_idx = tdata.obs_names.get_loc("F")
    # Path metric: path(C,D) = |3+3-2*2| = 2; path(C,F) = |3+3-2*0| = 6
    tree_neighbors(tdata, max_dist=10, metric="path", tree="tree1")
    dist = tdata.obsp["tree_distances"]
    assert dist[c_idx, d_idx] == 2
    assert dist[c_idx, f_idx] == 6
    # LCA metric (default depth key): LCA(C,D)=A depth=2; LCA(C,F)=root depth=0
    tree_neighbors(tdata, max_dist=10, metric="lca", tree="tree1", key_added="lca")
    lca_dist = tdata.obsp["lca_distances"]
    assert lca_dist[c_idx, d_idx] == 2
    assert lca_dist[c_idx, f_idx] == 0


def test_tree_neighbors_invalid(tdata):
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="invalid")  # type: ignore
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="path", obs="invalid")
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="path", tree="invalid")
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="path", tree=["tree1", "invalid"])
    with pytest.raises(KeyError):
        tree_neighbors(tdata, n_neighbors=3, metric="path", obs=["C", "invalid"])
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="path", depth_key="invalid")


@pytest.fixture
def nodes_tdata():
    # Tree:  root(0) -> A(1), B(1); A -> C(2), D(2); B -> E(2)
    tree = nx.DiGraph([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("B", "E")])
    nx.set_node_attributes(tree, {"root": 0, "A": 1, "B": 1, "C": 2, "D": 2, "E": 2}, "depth")
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["root", "A", "B", "C", "D", "E"]),
        obst={"tree": tree},
        alignment="nodes",
    )
    return tdata


def test_tree_neighbors_nodes_alignment(nodes_tdata):
    # path(A, C) = |1+2 - 2*1| = 1; path(A, B) = |1+1 - 2*0| = 2; path(C, E) = |2+2 - 2*0| = 4
    tree_neighbors(nodes_tdata, max_dist=2, metric="path")
    dist = nodes_tdata.obsp["tree_distances"]
    a_idx = nodes_tdata.obs_names.get_loc("A")
    b_idx = nodes_tdata.obs_names.get_loc("B")
    c_idx = nodes_tdata.obs_names.get_loc("C")
    d_idx = nodes_tdata.obs_names.get_loc("D")
    # Internal nodes appear as neighbors
    assert dist[a_idx, c_idx] == 1  # A -> C (path = 1)
    assert dist[a_idx, d_idx] == 1  # A -> D (path = 1)
    assert dist[c_idx, a_idx] == 1  # C -> A (path = 1)
    assert dist[a_idx, b_idx] == 2  # A -> B (path = 2)
    # matrix is n_obs x n_obs (6 observations including internal nodes)
    assert dist.shape == (6, 6)
    # LCA metric: lca(A, C) = A (depth 1); lca(C, D) = A (depth 1); lca(C, E) = root (depth 0)
    tree_neighbors(nodes_tdata, max_dist=2, metric="lca", key_added="lca")
    lca_dist = nodes_tdata.obsp["lca_distances"]
    assert lca_dist[a_idx, c_idx] == 1  # lca(A, C) = A, depth = 1
    assert lca_dist[c_idx, d_idx] == 1  # lca(C, D) = A, depth = 1


def test_tree_neighbors_nodes_single_obs(nodes_tdata):
    # For a single-string obs, only the queried node is marked True (consistent behavior)
    tree_neighbors(nodes_tdata, n_neighbors=3, metric="path", obs="A")
    assert nodes_tdata.obs.query("tree_neighbors").index.tolist() == ["A"]
    # Internal node (B) is also a valid starting point
    tree_neighbors(nodes_tdata, n_neighbors=2, metric="path", obs="B")
    assert nodes_tdata.obs.query("tree_neighbors").index.tolist() == ["B"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
