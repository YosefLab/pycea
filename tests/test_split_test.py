import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.tl import split_test
from pycea.utils import _get_descendant_leaves, get_leaves

# -------------------------
# Helpers / fixtures
# -------------------------


@pytest.fixture
def deep_balanced_tdata():
    """
    Perfect binary tree of depth 10 (1024 leaves) under obst['balanced'].
    obs index are leaf names only, matching default 'alignment="obs"'.
    """
    tree = nx.balanced_tree(2, 10, nx.DiGraph)
    nx.relabel_nodes(tree, {i: str(i) for i in range(len(tree.nodes()))}, copy=False)
    # Start with an empty obs; tests will fill values per scenario
    obs = pd.DataFrame(index=get_leaves(tree))
    tdata = td.TreeData(obs=obs, obst={"balanced": tree})
    yield tdata


def _attach_k_leaves(G: nx.DiGraph, parent: str, k: int, prefix: str) -> list[str]:
    leaves = []
    for i in range(k):
        leaf = f"{prefix}_{i}"
        G.add_edge(parent, leaf)
        leaves.append(leaf)
    return leaves


def _three_way_root_tree(k_per_child: int = 100) -> nx.DiGraph:
    """
    root has exactly three children: 'left', 'middle', 'right'
    each with k_per_child leaf children.
    """
    G = nx.DiGraph()
    G.add_node("root")
    for child in ("left", "middle", "right"):
        G.add_edge("root", child)
        _attach_k_leaves(G, child, k_per_child, prefix=f"{child}_leaf")
    return G


def _obs_for_three_way(G: nx.DiGraph, left_val: float, middle_val: float, right_val: float) -> pd.DataFrame:
    rows = []
    for child, val in (("left", left_val), ("middle", middle_val), ("right", right_val)):
        for leaf in G.successors(child):  # all direct leaves
            rows.append((leaf, val))
    return pd.DataFrame({"x": [v for _, v in rows]}, index=[n for n, _ in rows])


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
    root = "0"
    children = list(t.successors(root))
    assert len(children) == 2
    left_child, right_child = children

    leaves_dict = _get_descendant_leaves(t)

    left_desc_leaves = leaves_dict[left_child]
    right_desc_leaves = leaves_dict[right_child]

    # Assign values: left = 1, right = 0
    obs = pd.DataFrame({"x": 1}, index=left_desc_leaves)
    obs_right = pd.DataFrame({"x": 0}, index=right_desc_leaves)
    deep_balanced_tdata.obs = pd.concat([obs, obs_right]).sort_index()

    # Run with a modest number of permutations to keep the test fast
    n_perms = 100
    states = split_test(
        deep_balanced_tdata,
        keys="x",
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=n_perms,
        tree="balanced",
        copy=True,
    )
    # Returned DataFrame checks
    assert states is not None

    # Edge attributes at the root's children
    assert t.nodes[left_child]["x_value"] == 1.0
    assert t.nodes[right_child]["x_value"] == 0.0

    # Node attributes (via returned DataFrame)
    pval = t[root][right_child]["x_pval"]
    # With all 1s vs all 0s, the two-sided p-value under permutations should be ~ 1/(n_perms+1)
    # so we assert it's less than 1/(n_perms)
    assert pval <= (1 / n_perms)

    # run permutations vs. rest
    states_vs_rest = split_test(
        deep_balanced_tdata,
        keys="x",
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=n_perms,
        tree="balanced",
        copy=True,
        comparison="rest",
    )

    assert states_vs_rest is not None
    assert t.nodes[left_child]["x_value"] == 1.0
    assert t.nodes[right_child]["x_value"] == 0.0
    pval = t[root][right_child]["x_pval"]
    assert pval <= (1 / n_perms)
    assert 2 * states.shape[0] == states_vs_rest.shape[0]

    # make sure that results stay the same if a few missing y values are inserted

    # Start from the existing 'x' values to define 'y'
    y = deep_balanced_tdata.obs["x"].copy()

    # Pick a few indices on each side to set as NaN (keep deterministic selection)
    left_sorted = sorted(left_desc_leaves)
    right_sorted = sorted(right_desc_leaves)
    # choose up to 3 from each side, but at least 1 if available
    left_na_idx = left_sorted[: min(3, len(left_sorted))] if len(left_sorted) > 0 else []
    right_na_idx = right_sorted[: min(3, len(right_sorted))] if len(right_sorted) > 0 else []

    # Inject NaNs
    y.loc[left_na_idx] = np.nan
    y.loc[right_na_idx] = np.nan

    # Add 'y' to obs (aligned by index)
    deep_balanced_tdata.obs["y"] = y
    states_xy = split_test(
        deep_balanced_tdata,
        keys=["x", "y"],
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=n_perms,
        tree="balanced",
        copy=True,
    )
    assert states_xy is not None
    assert states_xy.shape[0] == 2040
    assert t.nodes[left_child]["x_value"] == 1.0
    assert t.nodes[right_child]["x_value"] == 0.0

    # Node attributes (via returned DataFrame)
    pval = t[root][right_child]["x_pval"]
    assert pval <= (1 / n_perms)


def test_split_permutation_root_null_case(deep_balanced_tdata):
    """
    Null case: all leaves = 1. Expect left_stat=1, right_stat=1, split_stat=0, p-value = 1.0.
    """
    t = deep_balanced_tdata.obst["balanced"]

    # All leaves get value 1
    leaves = get_leaves(t)
    deep_balanced_tdata.obs = pd.DataFrame({"x": 1}, index=leaves)

    # Run
    states = split_test(
        deep_balanced_tdata,
        keys="x",
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=10,  # any value; distribution is degenerate
        tree="balanced",
        copy=True,
    )
    assert states is not None

    root = "0"
    children = list(t.successors(root))
    left_child, right_child = children

    # Edge attributes: both 1
    assert t.nodes[left_child]["x_value"] == 1.0
    assert t.nodes[right_child]["x_value"] == 1.0

    pval = t[root][right_child]["x_pval"]
    assert pytest.approx(pval, rel=0, abs=1e-12) == 1.0


def test_nonbinary_positive_control_one_vs_rest_small_p():
    """
    Positive control:
      - left=1, middle=1, right=0
      - One-vs-rest at root:
          * 'right' vs rest -> 0 vs mean(left+middle)=1 => large observed effect
      Expect the p-value for 'right' to be very small.
    """
    G = _three_way_root_tree(k_per_child=100)
    obs = _obs_for_three_way(G, left_val=1.0, middle_val=1.0, right_val=0.0)

    tdata = td.TreeData(obs=obs, obst={"tri": G})
    states = split_test(
        tdata,
        keys="x",
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=100,
        tree="tri",
        copy=True,
    )
    assert states is not None

    # Pick the row for the root where group1 == 'right' (right vs rest)
    root_rows = states[(states["tree"] == "tri") & (states["parent"] == "root")]
    row_right = root_rows[root_rows["group1"] == "right"].iloc[0]
    pval = row_right["pval"]
    assert pval <= 0.01  # generous but should be clearly small


def test_nonbinary_negative_control_middle_vs_rest_p_near_one():
    """
    Negative control:
      - left=1, middle=0.5, right=0
      - One-vs-rest at root:
          * 'middle' vs rest -> 0.5 vs mean(left,right)=(1+0)/2 = 0.5 -> observed effect = 0
      Expect the p-value for 'middle' to be ~1.
    """
    G = _three_way_root_tree(k_per_child=100)
    obs = _obs_for_three_way(G, left_val=1.0, middle_val=0.5, right_val=0.0)

    tdata = td.TreeData(obs=obs, obst={"tri": G})
    states = split_test(
        tdata,
        keys="x",
        aggregate="mean",
        metric="mean_difference",
        test="permutation",
        n_permutations=100,
        tree="tri",
        copy=True,
    )
    assert states is not None

    # Pick the row for the root where group1 == 'middle' (middle vs rest)
    root_rows = states[(states["tree"] == "tri") & (states["parent"] == "root")]
    row_middle = root_rows[root_rows["group1"] == "middle"].iloc[0]
    pval = row_middle["pval"]
    assert pytest.approx(pval, rel=0, abs=1e-12) == 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
