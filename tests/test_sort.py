import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl.sort import sort


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "B"), ("root", "C")])
    nx.set_node_attributes(tree1, {"root": 1, "B": 3, "C": 2}, "value")
    nx.set_edge_attributes(
        tree1, {("root", "B"): {"length": 1}, ("root", "C"): {"length": 2}}
    )

    tree2 = nx.DiGraph([("root", "D"), ("root", "E")])
    nx.set_node_attributes(tree2, {"root": "1", "D": "2", "E": "3"}, "str_value")
    nx.set_node_attributes(tree2, {"root": 1, "D": 2, "E": 3}, "value")
    nx.set_edge_attributes(
        tree2, {("root", "D"): {"length": 1}, ("root", "E"): {"length": 2}}
    )
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["B", "C", "D", "E"]),
        obst={"tree1": tree1, "tree2": tree2},
    )
    yield tdata


def test_sort(tdata):
    sort(tdata, "value", reverse=False)
    assert list(tdata.obst["tree1"].successors("root")) == ["C", "B"]
    assert list(tdata.obst["tree2"].successors("root")) == ["D", "E"]
    sort(tdata, "str_value", tree="tree2", reverse=True)
    assert list(tdata.obst["tree2"].successors("root")) == ["E", "D"]


def test_sort_invalid(tdata):
    with pytest.raises(KeyError):
        sort(tdata, "bad")


def test_sort_preserves_edge_metadata(tdata):
    tree1_lengths = {
        child: tdata.obst["tree1"]["root"][child]["length"]
        for child in tdata.obst["tree1"].successors("root")
    }
    tree2_lengths = {
        child: tdata.obst["tree2"]["root"][child]["length"]
        for child in tdata.obst["tree2"].successors("root")
    }

    sort(tdata, "value", reverse=False)
    sort(tdata, "str_value", tree="tree2", reverse=True)

    for child, length in tree1_lengths.items():
        assert tdata.obst["tree1"]["root"][child]["length"] == length
    for child, length in tree2_lengths.items():
        assert tdata.obst["tree2"]["root"][child]["length"] == length
