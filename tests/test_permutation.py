import numpy as np
import pandas as pd
import networkx as nx
import pytest

import treedata as td

from pycea.tl import split_permutation_test
from pycea.utils import get_leaves, get_leaves_from_node

# -------------------------
# Helpers / fixtures
# -------------------------

def _make_perfect_binary_tree(depth: int) -> nx.DiGraph:
    """
    Build a perfect binary tree of given depth (root at depth 0).
    Node names are strings like 'r', 'rL', 'rR', 'rLL', etc.
    Leaves are exactly the nodes at depth `depth`.
    """
    t = nx.DiGraph()
    root = "r"
    t.add_node(root)
    frontier = [root]
    for _ in range(depth):
        new_frontier = []
        for p in frontier:
            L = p + "L"
            R = p + "R"
            t.add_edge(p, L)
            t.add_edge(p, R)
            new_frontier.extend([L, R])
        frontier = new_frontier
    return t


@pytest.fixture
def deep_balanced_tdata():
    """
    Perfect binary tree of depth 10 (1024 leaves) under obst['balanced'].
    obs index are leaf names only, matching default 'alignment="obs"'.
    """
    tree = _make_perfect_binary_tree(depth=10)
    # Start with an empty obs; tests will fill values per scenario
    obs = pd.DataFrame(index=get_leaves(tree))
    tdata = td.TreeData(obs=obs, obst={"balanced": tree})
    yield tdata


@pytest.fixture
def tiny_tdata():
    """
    Tiny tree: root->L, root->R (two leaves). This guarantees
    comb(2,1)=2, so the permutation test should be skipped with default min_required_permutations=50.
    """
    tree = nx.DiGraph([("root", "L"), ("root", "R")])
    obs = pd.DataFrame({"value": [1, 0]}, index=["L", "R"])
    tdata = td.TreeData(obs=obs, obst={"tiny": tree})
    yield tdata


# -------------------------
# Tests
# -------------------------

def test_split_permutation_root_extreme_signal(deep_balanced_tdata):
    """
    Strong-signal case: all left leaves = 1, all right leaves = 0.
    Expect left_stat=1, right_stat=0 at root, split_stat=1, and a very small p-value.
    """
    t = deep_balanced_tdata.obst["balanced"]

    # Identify root, its two children, and their leaf sets
    root = "r"
    children = list(t.successors(root))
    assert len(children) == 2
    left_child, right_child = children

    left_desc_leaves = get_leaves_from_node(t, left_child)
    right_desc_leaves = get_leaves_from_node(t, right_child)

    # Assign values: left = 1, right = 0
    obs = pd.DataFrame({"value": 1}, index=left_desc_leaves)
    obs_right = pd.DataFrame({"value": 0}, index=right_desc_leaves)
    deep_balanced_tdata.obs = pd.concat([obs, obs_right]).sort_index()

    # Run with a modest number of permutations to keep the test fast
    n_perms = 100
    states = split_permutation_test(
        deep_balanced_tdata,
        keys="value",
        reduction_fn=np.mean,
        permutation_test=True,
        n_permutations=n_perms,
        keys_added="value",
        tree="balanced",
        copy=True,
    )
    # Returned DataFrame checks
    assert states is not None
    assert ("balanced", root) in states.index
    assert "value_split" in states.columns and "value_pval" in states.columns

    # Edge attributes at the root's children
    assert t[root][left_child]["value"] == 1.0
    assert t[root][right_child]["value"] == 0.0

    # Node attributes (via returned DataFrame)
    split_val = states.loc[("balanced", root), "value_split"]
    pval = states.loc[("balanced", root), "value_pval"]
    assert pytest.approx(split_val, rel=0, abs=1e-12) == 1.0
    # With all 1s vs all 0s, the two-sided p-value under permutations should be ~ 1/(n_perms+1)
    # so we assert it's less than 1/(n_perms)
    assert pval <= (1 / n_perms)


def test_split_permutation_root_null_case(deep_balanced_tdata):
    """
    Null case: all leaves = 1. Expect left_stat=1, right_stat=1, split_stat=0, p-value = 1.0.
    """
    t = deep_balanced_tdata.obst["balanced"]

    # All leaves get value 1
    leaves = get_leaves(t)
    deep_balanced_tdata.obs = pd.DataFrame({"value": 1}, index=leaves)

    # Run
    states = split_permutation_test(
        deep_balanced_tdata,
        keys="value",
        reduction_fn=np.mean,
        permutation_test=True,
        n_permutations=10,  # any value; distribution is degenerate
        min_required_permutations=5,
        keys_added="value",
        tree="balanced",
        copy=True,
    )
    assert states is not None

    root = "r"
    children = list(t.successors(root))
    left_child, right_child = children

    # Edge attributes: both 1
    assert t[root][left_child]["value"] == 1.0
    assert t[root][right_child]["value"] == 1.0

    # Node attributes: split=0, pval=1.0 (degenerate permutations)
    split_val = states.loc[("balanced", root), "value_split"]
    pval = states.loc[("balanced", root), "value_pval"]
    assert pytest.approx(split_val, rel=0, abs=1e-12) == 0.0
    assert pytest.approx(pval, rel=0, abs=1e-12) == 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
