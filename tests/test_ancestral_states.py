from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.tl.ancestral_states import ancestral_states


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "B"), ("root", "C"), ("C", "D"), ("C", "E")])
    tree2 = nx.DiGraph([("root", "F")])
    spatial = np.array([[0, 4], [1, 1], [2, 1], [4, 4]])
    characters = np.array([["-1", "0"], ["1", "1"], ["2", "-1"], ["1", "2"]])
    tdata = td.TreeData(
        obs=pd.DataFrame(
            {"value": [0, 0, 3, 2], "str_value": ["0", "0", "3", "2"], "with_missing": [0, np.nan, 3, 2]},
            index=["B", "D", "E", "F"],
        ),
        obst={"tree1": tree1, "tree2": tree2},
        obsm={"spatial": spatial, "characters": characters},
    )
    yield tdata


def test_ancestral_states(tdata):
    # Mean
    states = ancestral_states(tdata, "value", method="mean", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 1
    assert tdata.obst["tree1"].nodes["C"]["value"] == 1.5
    assert states["value"].tolist() == [1, 0, 1.5, 0, 3, 2, 2]
    # Median
    states = ancestral_states(tdata, "value", method=np.median, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 0
    # Mode
    ancestral_states(tdata, "str_value", method="mode", copy=False, tree="tree1")
    for node in tdata.obst["tree1"].nodes:
        print(node, tdata.obst["tree1"].nodes[node])
    assert tdata.obst["tree1"].nodes["root"]["str_value"] == "0"


def test_ancestral_states_array(tdata):
    # Mean
    states = ancestral_states(tdata, "spatial", method="mean", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["spatial"] == [1.0, 2.0]
    assert tdata.obst["tree1"].nodes["C"]["spatial"] == [1.5, 1.0]
    assert states.loc[("tree1", "root"), "spatial"] == [1.0, 2.0]
    # Median
    states = ancestral_states(tdata, "spatial", method=np.median, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["spatial"] == [1.0, 1.0]


def test_ancestral_states_missing(tdata):
    # Mean
    states = ancestral_states(tdata, "with_missing", method=np.nanmean, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["with_missing"] == 1.5
    assert tdata.obst["tree1"].nodes["C"]["with_missing"] == 3
    assert states.loc[("tree1", "root"), "with_missing"] == 1.5


def test_ancestral_state_fitch(tdata):
    states = ancestral_states(tdata, "characters", method="fitch_hartigan", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["1", "0"]
    assert tdata.obst["tree2"].nodes["F"]["characters"] == ["1", "2"]
    assert states.loc[("tree1", "root"), "characters"] == ["1", "0"]


def test_ancestral_states_sankoff(tdata):
    costs = pd.DataFrame(
        [[0, 1, 2], [10, 0, 10], [10, 10, 0]],
        index=["0", "1", "2"],
        columns=["0", "1", "2"],
    )
    states = ancestral_states(tdata, "characters", method="sankoff", costs=costs, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["0", "0"]
    assert tdata.obst["tree2"].nodes["F"]["characters"] == ["1", "2"]
    assert states.loc[("tree1", "root"), "characters"] == ["0", "0"]
    costs = pd.DataFrame(
        [[0, 10, 10], [1, 0, 2], [2, 1, 0]],
        index=["0", "1", "2"],
        columns=["0", "1", "2"],
    )
    states = ancestral_states(tdata, "characters", method="sankoff", costs=costs, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["2", "1"]


def test_ancestral_states_invalid(tdata):
    with pytest.raises(ValueError):
        ancestral_states(tdata, "characters", method="sankoff")
    with pytest.raises((ValueError, KeyError)):
        ancestral_states(tdata, "characters", method="sankoff", costs=pd.DataFrame())
    with pytest.raises(ValueError):
        ancestral_states(tdata, "bad", method="mean")
    with pytest.raises(ValueError):
        ancestral_states(tdata, "value", method="bad")
    with pytest.raises(ValueError):
        ancestral_states(tdata, "str_value", method="mean", copy=False)
