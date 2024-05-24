import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl.ancestral_states import ancestral_states


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "B"), ("root", "C"), ("C", "D"), ("C", "E")])
    tree2 = nx.DiGraph([("root", "F")])
    tdata = td.TreeData(
        obs=pd.DataFrame({"value": [0, 0, 3, 2], "str_value": ["0", "0", "3", "2"]}, index=["B", "D", "E", "F"]),
        obst={"tree1": tree1, "tree2": tree2},
    )
    yield tdata


def test_ancestral_states(tdata):
    # Mean
    states = ancestral_states(tdata, "value", method="mean", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 1
    assert tdata.obst["tree1"].nodes["C"]["value"] == 1.5
    assert states["value"].tolist() == [1, 0, 1.5, 0, 3, 2, 2]
    # Median
    states = ancestral_states(tdata, "value", method="median", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 0
    # Mode
    ancestral_states(tdata, "str_value", method="mode", copy=False, tree="tree1")
    assert tdata.obst["tree1"].nodes["root"]["str_value"] == "0"
