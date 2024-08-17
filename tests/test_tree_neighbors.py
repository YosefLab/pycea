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
    (dist, neighbors) = tree_neighbors(tdata, max_dist=3, metric="path", copy=True)
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
    assert tdata.obs.query("tree_neighbors").index.tolist() == ["D", "E"]
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


def test_tree_neighbors_invalid(tdata):
    with pytest.raises(ValueError):
        tree_neighbors(tdata, n_neighbors=3, metric="invalid")
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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
