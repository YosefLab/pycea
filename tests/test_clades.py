import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.tl.clades import _nodes_at_depth, clades


@pytest.fixture
def tree():
    t = nx.DiGraph()
    t.add_edges_from([("A", "B"), ("A", "C"), ("C", "D"), ("C", "E")])
    nx.set_node_attributes(t, {"A": 0, "B": 2, "C": 1, "D": 2, "E": 2}, "depth")
    yield t


@pytest.fixture
def tdata(tree):
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "D", "E"]), obst={"tree": tree, "empty": nx.DiGraph()})
    yield tdata


@pytest.fixture
def nodes_tdata(tree):
    nodes_tdata = td.TreeData(obs=pd.DataFrame(index=["A", "B", "C", "D", "E"]), obst={"tree": tree}, alignment="nodes")
    yield nodes_tdata


def test_nodes_at_depth(tree):
    assert _nodes_at_depth(tree, "A", [], 0, "depth") == ["A"]
    assert _nodes_at_depth(tree, "A", [], 1, "depth") == ["B", "C"]
    assert _nodes_at_depth(tree, "A", [], 2, "depth") == ["B", "D", "E"]


def test_clades_given_dict(tdata):
    clades(tdata, clades={"B": 0, "C": 1})
    assert tdata.obs["clade"].tolist() == [0, 1, 1]
    assert tdata.obst["tree"].nodes["C"]["clade"] == 1
    assert tdata.obst["tree"].edges[("C", "D")]["clade"] == 1
    clades(tdata, clades={"A": "0"}, key_added="all")
    assert tdata.obs["all"].tolist() == ["0", "0", "0"]
    assert tdata.obst["tree"].nodes["A"]["all"] == "0"
    assert tdata.obst["tree"].edges[("C", "D")]["all"] == "0"


def test_clades_given_depth(tdata):
    clades(tdata, depth=0)
    assert tdata.obs["clade"].tolist() == ["0", "0", "0"]
    clade_nodes = clades(tdata, depth=1, copy=True)
    assert clade_nodes["node"].tolist() == ["B", "C"]
    assert clade_nodes["clade"].tolist() == ["0", "1"]
    assert tdata.obs["clade"].tolist() == ["0", "1", "1"]
    clades(tdata, depth=2)
    assert tdata.obs["clade"].tolist() == ["0", "1", "2"]


def test_clades_update(tdata):
    clades(tdata, depth=0)
    assert tdata.obs["clade"].tolist() == ["0", "0", "0"]
    clades(tdata, depth=1, update=True)
    assert tdata.obst["tree"].nodes["A"]["clade"] == "0"
    assert tdata.obs["clade"].tolist() == ["0", "1", "1"]
    clades(tdata, depth=1, update=False)
    assert "clade" not in tdata.obst["tree"].nodes["A"]
    assert tdata.obs["clade"].tolist() == ["0", "1", "1"]
    clades(tdata, clades={"D": "2"}, update=True)
    assert tdata.obs["clade"].tolist() == ["0", "2", "1"]


def test_clades_multiple_trees():
    tree1 = nx.DiGraph([("root", "A")])
    nx.set_node_attributes(tree1, {"root": 0, "A": 1}, "depth")
    tree2 = nx.DiGraph([("root", "B")])
    nx.set_node_attributes(tree2, {"root": 0, "B": 2}, "depth")
    tdata = td.TreeData(obs=pd.DataFrame(index=["A", "B"]), obst={"tree1": tree1, "tree2": tree2})
    clades(tdata, depth=0)
    assert tdata.obs["clade"].tolist() == ["0", "1"]
    # need to specify tree with clade input
    with pytest.raises(ValueError):
        clades(tdata, clades={"root": 0})
    clades(tdata, clades={"root": "0"}, tree="tree1", key_added="test")
    assert tdata.obs.loc["A", "test"] == "0"
    assert pd.isna(tdata.obs.loc["B", "test"])


def test_clades_nodes_tdata(nodes_tdata):
    clades(nodes_tdata, depth=1)
    assert nodes_tdata.obs["clade"].tolist() == [np.nan, "0", "1", "1", "1"]


def test_clades_dtype(tdata):
    clades(tdata, depth=0, dtype=int)
    assert tdata.obs["clade"].dtype == int
    assert tdata.obst["tree"].nodes["A"]["clade"] == 0
    clades(tdata, depth=0, dtype="int")
    assert tdata.obs["clade"].dtype == int
    assert tdata.obst["tree"].nodes["A"]["clade"] == 0
    clades(tdata, depth=1, dtype=float)
    assert tdata.obs["clade"].dtype == float
    assert tdata.obst["tree"].nodes["C"]["clade"] == 1.0


def test_clades_invalid(tdata):
    with pytest.raises(ValueError):
        clades(td.TreeData(), clades={"A": 0}, depth=0)
    with pytest.raises(ValueError):
        clades(tdata, clades={"A": 0}, depth=0)
    with pytest.raises((KeyError, nx.NetworkXError)):
        clades(tdata, clades={"bad": 0}, key_added="clade")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
