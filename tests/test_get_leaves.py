import networkx as nx
import treedata as td

from pycea.get import leaves


def _make_tree(edges):
    t = nx.DiGraph()
    t.add_edges_from(edges)
    return t


def test_leaves_single_tree():
    t1 = _make_tree([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    tdata = td.TreeData(obst={"t1": t1})
    assert leaves(tdata) == ["D", "E"]
    assert leaves(tdata, "t1") == ["D", "E"]


def test_leaves_multiple_trees():
    t1 = _make_tree([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    t2 = _make_tree([("F", "G"), ("F", "H")])
    tdata = td.TreeData(obst={"t1": t1, "t2": t2})
    res_all = leaves(tdata)
    assert res_all == {"t1": ["D", "E"], "t2": ["G", "H"]}
    res_subset = leaves(tdata, ["t1", "t2"])
    assert res_subset == {"t1": ["D", "E"], "t2": ["G", "H"]}
    assert leaves(tdata, "t2") == ["G", "H"]
