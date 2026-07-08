from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl.ancestral_states import ancestral_states
from pycea.tl.parsimony import fitch_count, parsimony


@pytest.fixture
def tdata():
    # tree1: root -> A, B; B -> C, D   (leaves A, C, D)
    # tree2: root -> E, F              (leaves E, F)
    tree1 = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D")])
    tree2 = nx.DiGraph([("root", "E"), ("root", "F")])
    tdata = td.TreeData(
        obs=pd.DataFrame(
            {"clone": ["x", "x", "y", "x", "y"]},
            index=["A", "C", "D", "E", "F"],
        ),
        obst={"tree1": tree1, "tree2": tree2},
    )
    yield tdata


def test_parsimony_reconstruct(tdata):
    score = parsimony(tdata, "clone", tree="tree1", copy=True)
    assert score == 1
    assert tdata.uns["parsimony"] == 1
    # states were reconstructed on the tree
    assert tdata.obst["tree1"].nodes["root"]["clone"] == "x"


def test_parsimony_multiple_trees(tdata):
    scores = parsimony(tdata, "clone", copy=True)
    assert scores.loc["tree1"] == 1
    assert scores.loc["tree2"] == 1


def test_parsimony_no_reconstruct(tdata):
    ancestral_states(tdata, "clone", method="fitch_hartigan", tree="tree1")
    score = parsimony(tdata, "clone", reconstruct=False, tree="tree1", copy=True)
    assert score == 1


def test_parsimony_no_reconstruct_missing_key(tdata):
    with pytest.raises(ValueError):
        parsimony(tdata, "clone", reconstruct=False, tree="tree1")


def test_fitch_count_single_tree(tdata):
    M = fitch_count(tdata, "clone", tree="tree1", copy=True)
    assert isinstance(M, pd.DataFrame)
    assert M.loc["x", "x"] == 3
    assert M.loc["x", "y"] == 1
    assert M.loc["y", "x"] == 0
    assert M.loc["y", "y"] == 0
    # stored in uns
    assert isinstance(tdata.uns["fitch_count"], pd.DataFrame)


def test_fitch_count_two_mp_solutions(tdata):
    M = fitch_count(tdata, "clone", tree="tree2", copy=True)
    # root has both x and y in its Fitch set -> two equally parsimonious solutions
    assert M.loc["x", "x"] == 1
    assert M.loc["x", "y"] == 1
    assert M.loc["y", "x"] == 1
    assert M.loc["y", "y"] == 1


def test_fitch_count_multiple_trees(tdata):
    # tree1: [[3,1],[0,0]] + tree2: [[1,1],[1,1]] summed across trees
    result = fitch_count(tdata, "clone", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert result.loc["x", "x"] == 4
    assert result.loc["x", "y"] == 2
    assert result.loc["y", "x"] == 1
    assert result.loc["y", "y"] == 1


def test_fitch_count_explicit_states(tdata):
    M = fitch_count(tdata, "clone", states=["x", "y", "z"], tree="tree1", copy=True)
    assert list(M.index) == ["x", "y", "z"]
    assert M.loc["x", "x"] == 3
    assert M.loc["z"].sum() == 0


def test_fitch_count_invalid_states(tdata):
    with pytest.raises(ValueError):
        fitch_count(tdata, "clone", states=["x"], tree="tree1")


def test_fitch_count_root_requires_single_tree(tdata):
    with pytest.raises(ValueError):
        fitch_count(tdata, "clone", root="B")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
