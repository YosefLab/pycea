import networkx as nx
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
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "D", "E"]), obst={"tree": tree})
    yield tdata


def test_nodes_at_depth(tree):
    assert _nodes_at_depth(tree, "A", [], 0, "depth") == ["A"]
    assert _nodes_at_depth(tree, "A", [], 1, "depth") == ["B", "C"]
    assert _nodes_at_depth(tree, "A", [], 2, "depth") == ["B", "D", "E"]


def test_clades_given_dict(tdata, tree):
    clades(tdata, clades={"B": 0, "C": 1})
    assert tdata.obs["clade"].tolist() == [0, 1, 1]
    assert tdata.obst["tree"].nodes["C"]["clade"] == 1
    assert tdata.obst["tree"].edges[("C", "D")]["clade"] == 1
    clades(tdata, clades={"A": "0"}, clade_key="all")
    assert tdata.obs["all"].tolist() == ["0", "0", "0"]
    assert tdata.obst["tree"].nodes["A"]["all"] == "0"
    assert tdata.obst["tree"].edges[("C", "D")]["all"] == "0"


def test_clades_given_depth(tdata):
    clades(tdata, depth=0)
    assert tdata.obs["clade"].tolist() == ["0", "0", "0"]
    nodes = clades(tdata, depth=1, copy=True)
    assert tdata.obs["clade"].tolist() == ["0", "1", "1"]
    assert nodes == {"B": "0", "C": "1"}
    clades(tdata, depth=2)
    assert tdata.obs["clade"].tolist() == ["0", "1", "2"]
