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
        obst={"tree1": tree1, "tree2": tree2, "empty": nx.DiGraph()},
        obsm={"spatial": spatial, "characters": characters},  # type: ignore
    )
    yield tdata


@pytest.fixture
def nodes_tdata():
    # Tree: root -> B, C; C -> D, E
    # spatial: root=NaN (reconstruct), B=[0,1] (leaf), C=[1,1] (fixed), D=[2,1] (leaf), E=[4,4] (leaf)
    tree = nx.DiGraph([("root", "B"), ("root", "C"), ("C", "D"), ("C", "E")])
    spatial = np.array([[np.nan, np.nan], [0, 1], [1, 1], [2, 1], [4, 4]])
    nodes_tdata = td.TreeData(
        obs=pd.DataFrame(
            {"value": [np.nan, 0, 5, 3, 2], "str_value": [None, "0", "1", "3", "2"]},
            index=["root", "B", "C", "D", "E"],
        ),
        obst={"tree": tree},
        obsm={"spatial": spatial},  # type: ignore
        alignment="nodes",
    )
    yield nodes_tdata


def test_ancestral_states(tdata):
    # Mean
    states = ancestral_states(tdata, "value", method="mean", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 1
    assert tdata.obst["tree1"].nodes["C"]["value"] == 1.5
    assert states is not None
    assert states["value"].tolist() == [1, 0, 1.5, 0, 3, 2, 2]
    # Median
    states = ancestral_states(tdata, "value", method=np.median, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["value"] == 0
    # Mode
    ancestral_states(
        tdata,
        ["value", "str_value"],
        method="mode",
        copy=False,
        tree="tree1",
        keys_added=["value_mode", "str_value_mode"],
    )
    for node in tdata.obst["tree1"].nodes:
        print(node, tdata.obst["tree1"].nodes[node])
    assert tdata.obst["tree1"].nodes["root"]["str_value_mode"] == "0"
    assert tdata.obst["tree1"].nodes["root"]["value_mode"] == 0


def test_ancestral_states_array(tdata):
    # Mean
    states = ancestral_states(tdata, "spatial", method="mean", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["spatial"] == [1.0, 2.0]
    assert tdata.obst["tree1"].nodes["C"]["spatial"] == [1.5, 1.0]
    assert states is not None
    assert states.loc[("tree1", "root"), "spatial"] == [1.0, 2.0]
    # Median
    states = ancestral_states(tdata, "spatial", method=np.median, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["spatial"] == [1.0, 1.0]


def test_ancestral_states_missing(tdata):
    # Mean
    states = ancestral_states(tdata, "with_missing", method=np.nanmean, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["with_missing"] == 1.5
    assert tdata.obst["tree1"].nodes["C"]["with_missing"] == 3
    assert states is not None
    assert states.loc[("tree1", "root"), "with_missing"] == 1.5


def test_ancestral_state_fitch(tdata):
    states = ancestral_states(tdata, "characters", method="fitch_hartigan", missing_state="-1", copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["1", "0"]
    assert tdata.obst["tree2"].nodes["F"]["characters"] == ["1", "2"]
    assert states is not None
    assert states.loc[("tree1", "root"), "characters"] == ["1", "0"]


def test_ancestral_states_sankoff(tdata):
    costs = pd.DataFrame(
        [[0, 1, 2], [10, 0, 10], [10, 10, 0]],
        index=["0", "1", "2"],
        columns=["0", "1", "2"],
    )
    states = ancestral_states(tdata, "characters", method="sankoff", missing_state="-1", costs=costs, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["0", "0"]
    assert tdata.obst["tree2"].nodes["F"]["characters"] == ["1", "2"]
    assert states is not None
    assert states.loc[("tree1", "root"), "characters"] == ["0", "0"]
    costs = pd.DataFrame(
        [[0, 10, 10], [1, 0, 2], [2, 1, 0]],
        index=["0", "1", "2"],
        columns=["0", "1", "2"],
    )
    states = ancestral_states(tdata, "characters", method="sankoff", missing_state="-1", costs=costs, copy=True)
    assert tdata.obst["tree1"].nodes["root"]["characters"] == ["2", "1"]


def test_ancestral_states_nodes_tdata(nodes_tdata):
    # C=[1,1] is a fixed observed internal node; root has NaN (reconstructed)
    # root = mean(B=[0,1], C=[1,1]) = [0.5, 1.0]  (C treated as fixed, not expanded into D/E)
    states = ancestral_states(nodes_tdata, "spatial", method="mean", copy=True)
    assert nodes_tdata.obst["tree"].nodes["root"]["spatial"] == [0.5, 1.0]
    assert nodes_tdata.obst["tree"].nodes["C"]["spatial"] == [1, 1]  # C value preserved
    assert states.loc[("tree", "root"), "spatial"] == [0.5, 1.0]


def test_ancestral_states_nodes_scalar(nodes_tdata):
    # C=5 (fixed), root=NaN (reconstruct from B=0 and C=5)
    ancestral_states(nodes_tdata, "value", method="mean", copy=False)
    tree = nodes_tdata.obst["tree"]
    assert tree.nodes["C"]["value"] == 5  # C preserved
    assert tree.nodes["root"]["value"] == pytest.approx(2.5)  # mean(B=0, C=5)
    assert tree.nodes["B"]["value"] == 0  # leaf unchanged


def test_ancestral_states_nodes_fitch(nodes_tdata):
    # C="1" (fixed internal); root reconstructed from B="0" and C="1"
    ancestral_states(nodes_tdata, "str_value", method="fitch_hartigan", missing_state=None, copy=False)
    tree = nodes_tdata.obst["tree"]
    assert tree.nodes["C"]["str_value"] == "1"  # C value preserved


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
    with pytest.raises(ValueError):
        ancestral_states(tdata, "value", method="mean", keys_added=["bad", "bad"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
