import networkx as nx
import pytest
import treedata as td

from pycea.get import root


def _make_tree(edges):
    t = nx.DiGraph()
    t.add_edges_from(edges)
    return t


def test_root_single_tree():
    t1 = _make_tree([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    tdata = td.TreeData(obst={"t1": t1})
    assert root(tdata) == "A"
    assert root(tdata, "t1") == "A"


def test_root_multiple_trees():
    t1 = _make_tree([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])
    t2 = _make_tree([("F", "G"), ("F", "H")])
    tdata = td.TreeData(obst={"t1": t1, "t2": t2})
    res_all = root(tdata)
    assert res_all == {"t1": "A", "t2": "F"}
    res_subset = root(tdata, ["t1", "t2"])
    assert res_subset == {"t1": "A", "t2": "F"}
    assert root(tdata, "t1") == "A"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
