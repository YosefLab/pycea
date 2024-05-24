import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl.sort import sort


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "B"), ("root", "C")])
    nx.set_node_attributes(tree1, {"root": 1, "B": 3, "C": 2}, "value")
    tree2 = nx.DiGraph([("root", "D"), ("root", "E")])
    nx.set_node_attributes(tree2, {"root": "1", "D": "2", "E": "3"}, "str_value")
    nx.set_node_attributes(tree2, {"root": 1, "D": 2, "E": 3}, "value")
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "C", "D", "E"]), obst={"tree1": tree1, "tree2": tree2})
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
