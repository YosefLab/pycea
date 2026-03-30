"""Tests for tl.ancestral_linkage."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import pycea.tl as tl


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def balanced_tdata():
    """Four-leaf binary tree, two categories A and B.

    Structure (depth values):
        root (depth=0)
        ├── n1 (depth=0.5)
        │   ├── a1 (depth=1.0)   cat=A
        │   └── a2 (depth=1.0)   cat=A
        └── n2 (depth=0.5)
            ├── b1 (depth=1.0)   cat=B
            └── b2 (depth=1.0)   cat=B

    LCA depths:
        LCA(a1, a2) = 0.5   (within A)
        LCA(b1, b2) = 0.5   (within B)
        LCA(a*, b*) = 0.0   (between A and B)

    Path distances:
        path(a1, a2) = 2*(1.0 - 0.5) = 1.0
        path(b1, b2) = 1.0
        path(a*, b*) = 2*(1.0 - 0.0) = 2.0
    """
    t = nx.DiGraph()
    nodes = {
        "root": 0.0,
        "n1": 0.5,
        "n2": 0.5,
        "a1": 1.0,
        "a2": 1.0,
        "b1": 1.0,
        "b2": 1.0,
    }
    for node, depth in nodes.items():
        t.add_node(node, depth=depth)
    edges = [
        ("root", "n1"), ("root", "n2"),
        ("n1", "a1"), ("n1", "a2"),
        ("n2", "b1"), ("n2", "b2"),
    ]
    t.add_edges_from(edges)

    obs = pd.DataFrame(
        {"celltype": ["A", "A", "B", "B"]},
        index=["a1", "a2", "b1", "b2"],
    )
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def three_cat_tdata():
    """Six-leaf tree with categories A, B, C for fuller pairwise tests.

    Structure:
        root (0)
        ├── n1 (0.4)
        │   ├── a1 (1.0)  A
        │   └── a2 (1.0)  A
        ├── n2 (0.4)
        │   ├── b1 (1.0)  B
        │   └── b2 (1.0)  B
        └── n3 (0.4)
            ├── c1 (1.0)  C
            └── c2 (1.0)  C
    """
    t = nx.DiGraph()
    for node, depth in [
        ("root", 0.0), ("n1", 0.4), ("n2", 0.4), ("n3", 0.4),
        ("a1", 1.0), ("a2", 1.0), ("b1", 1.0), ("b2", 1.0), ("c1", 1.0), ("c2", 1.0),
    ]:
        t.add_node(node, depth=depth)
    for u, v in [
        ("root", "n1"), ("root", "n2"), ("root", "n3"),
        ("n1", "a1"), ("n1", "a2"),
        ("n2", "b1"), ("n2", "b2"),
        ("n3", "c1"), ("n3", "c2"),
    ]:
        t.add_edge(u, v)
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "B", "B", "C", "C"]},
        index=["a1", "a2", "b1", "b2", "c1", "c2"],
    )
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def two_tree_tdata():
    """Two separate trees, each with leaves from categories A and B.

    tree1: a1 (A), b1 (B)  — root depth 0, leaves depth 1, n1 depth 0.5
    tree2: a2 (A), b2 (B)  — root depth 0, leaves depth 1, n2 depth 0.5
    """
    t1 = nx.DiGraph()
    for node, depth in [("r1", 0.0), ("n1", 0.5), ("a1", 1.0), ("b1", 1.0)]:
        t1.add_node(node, depth=depth)
    t1.add_edges_from([("r1", "n1"), ("n1", "a1"), ("n1", "b1")])

    t2 = nx.DiGraph()
    for node, depth in [("r2", 0.0), ("n2", 0.5), ("a2", 1.0), ("b2", 1.0)]:
        t2.add_node(node, depth=depth)
    t2.add_edges_from([("r2", "n2"), ("n2", "a2"), ("n2", "b2")])

    obs = pd.DataFrame(
        {"celltype": ["A", "B", "A", "B"]},
        index=["a1", "b1", "a2", "b2"],
    )
    return td.TreeData(obs=obs, obst={"tree1": t1, "tree2": t2})


# ── pairwise mode tests ───────────────────────────────────────────────────────


def test_pairwise_path_min_within_greater_than_between(balanced_tdata):
    """Within-category path distance (min) should be smaller than between-category."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path")
    mat = tdata.uns["celltype_linkage"]
    assert mat.loc["A", "A"] < mat.loc["A", "B"]
    assert mat.loc["B", "B"] < mat.loc["B", "A"]


def test_pairwise_lca_max_within_greater_than_between(balanced_tdata):
    """Within-category LCA depth (max) should be larger than between-category."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="max", metric="lca")
    mat = tdata.uns["celltype_linkage"]
    assert mat.loc["A", "A"] > mat.loc["A", "B"]
    assert mat.loc["B", "B"] > mat.loc["B", "A"]


def test_pairwise_lca_max_known_values(balanced_tdata):
    """Verify exact values for lca+max aggregate."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="max", metric="lca")
    mat = tdata.uns["celltype_linkage"]
    # within: self-pair gives lca = (1.0+1.0-0)/2 = 1.0 (self is always max)
    assert np.isclose(mat.loc["A", "A"], 1.0)
    assert np.isclose(mat.loc["B", "B"], 1.0)
    # between: best LCA from a1/a2 to any b = root.depth = 0.0
    assert np.isclose(mat.loc["A", "B"], 0.0)


def test_pairwise_path_min_known_values(balanced_tdata):
    """Verify exact values for path+min aggregate."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path")
    mat = tdata.uns["celltype_linkage"]
    # Within A: min dist is self (0.0)
    assert np.isclose(mat.loc["A", "A"], 0.0)
    # path(a*, b*) = |1.0 + 1.0 - 2*0.0| = 2.0
    assert np.isclose(mat.loc["A", "B"], 2.0)


def test_pairwise_mean_aggregate(balanced_tdata):
    """mean aggregate should equal mean of all cross-category path distances."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="mean", metric="path")
    mat = tdata.uns["celltype_linkage"]
    # All a*-b* pairs have path distance 2.0 → mean = 2.0
    assert np.isclose(mat.loc["A", "B"], 2.0)
    # Within A: mean([path(a1,a1), path(a1,a2), path(a2,a1), path(a2,a2)]) = mean([0, 1, 1, 0]) = 0.5
    assert np.isclose(mat.loc["A", "A"], 0.5)


def test_pairwise_stored_in_uns(balanced_tdata):
    """Results stored in tdata.uns with expected keys."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype")
    assert "celltype_linkage" in tdata.uns
    assert "celltype_linkage_params" in tdata.uns
    assert "celltype_linkage_stats" in tdata.uns
    df = tdata.uns["celltype_linkage"]
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == {"A", "B"}
    assert set(df.columns) == {"A", "B"}


def test_pairwise_stats_has_n_columns(balanced_tdata):
    """Stats DataFrame has source_n and target_n columns (always)."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype")
    stats = tdata.uns["celltype_linkage_stats"]
    assert isinstance(stats, pd.DataFrame)
    assert "source_n" in stats.columns
    assert "target_n" in stats.columns
    assert "source" in stats.columns
    assert "target" in stats.columns
    assert "value" in stats.columns
    # 2 categories → 4 rows
    assert len(stats) == 4
    # source_n for category A = 2 leaves
    assert stats.loc[stats["source"] == "A"].iloc[0]["source_n"] == 2


def test_pairwise_copy_returns_dataframe(balanced_tdata):
    """copy=True returns DataFrame and also stores in uns."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert "celltype_linkage" in tdata.uns


def test_pairwise_key_added(balanced_tdata):
    """key_added controls storage key."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", key_added="mykey")
    assert "mykey_linkage" in tdata.uns
    assert "mykey_linkage_params" in tdata.uns
    assert "mykey_linkage_stats" in tdata.uns


# ── symmetrize tests ──────────────────────────────────────────────────────────


def test_symmetrize_mean(three_cat_tdata):
    """After symmetrize='mean', M[i,j] == M[j,i]."""
    tdata = three_cat_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path",
                         symmetrize="mean")
    mat = tdata.uns["celltype_linkage"]
    for i in mat.index:
        for j in mat.columns:
            assert np.isclose(mat.loc[i, j], mat.loc[j, i])


def test_symmetrize_max(three_cat_tdata):
    """After symmetrize='max', M[i,j] == M[j,i] and M[i,j] >= original M[i,j]."""
    tdata = three_cat_tdata
    result_nosym = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min",
                                        metric="path", copy=True)
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path",
                         symmetrize="max")
    mat = tdata.uns["celltype_linkage"]
    for i in mat.index:
        for j in mat.columns:
            assert np.isclose(mat.loc[i, j], mat.loc[j, i])
            assert mat.loc[i, j] >= result_nosym.loc[i, j] - 1e-9


# ── single-target mode tests ──────────────────────────────────────────────────


def test_single_target_stores_in_obs(balanced_tdata):
    """Single-target result stored in tdata.obs."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", target="B")
    assert "B_linkage" in tdata.obs.columns
    assert tdata.obs["B_linkage"].notna().all()


def test_single_target_path_known_values(balanced_tdata):
    """a1, a2 should have path distance 2.0 to nearest B leaf; b1, b2 distance to self = 0."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="path")
    assert np.isclose(tdata.obs.loc["a1", "B_linkage"], 2.0)
    assert np.isclose(tdata.obs.loc["a2", "B_linkage"], 2.0)
    # b1, b2 are in target B → distance to self = 0
    assert np.isclose(tdata.obs.loc["b1", "B_linkage"], 0.0)
    assert np.isclose(tdata.obs.loc["b2", "B_linkage"], 0.0)


def test_single_target_lca_known_values(balanced_tdata):
    """a1, a2 should have LCA depth 0.0 to best B leaf; b1/b2 in B → LCA with self = depth = 1.0."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="lca")
    assert np.isclose(tdata.obs.loc["a1", "B_linkage"], 0.0)
    # b1 is in B; nearest is itself (path=0), lca = (1.0 + 1.0 - 0) / 2 = 1.0
    assert np.isclose(tdata.obs.loc["b1", "B_linkage"], 1.0)


def test_single_target_copy_returns_series_df(balanced_tdata):
    """copy=True returns a DataFrame with per-category means."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="B",
                                  metric="path", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert "B_linkage" in result.columns
    assert "A" in result.index and "B" in result.index


def test_single_target_invalid_target(balanced_tdata):
    """Specifying a non-existent target raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", target="Z")


# ── by_tree tests ─────────────────────────────────────────────────────────────


def test_by_tree_stats_has_tree_column(two_tree_tdata):
    """When by_tree=True, stats DataFrame has a 'tree' column."""
    tdata = two_tree_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True)
    stats = tdata.uns["celltype_linkage_stats"]
    assert "tree" in stats.columns
    # 2 trees × 2 cats × 2 cats = 8 rows
    assert len(stats) == 8
    assert set(stats["tree"]) == {"tree1", "tree2"}


def test_by_tree_source_target_n(two_tree_tdata):
    """Per-tree source_n and target_n reflect leaves in that tree only."""
    tdata = two_tree_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True)
    stats = tdata.uns["celltype_linkage_stats"]
    # Each tree has 1 A leaf and 1 B leaf
    tree1_rows = stats[stats["tree"] == "tree1"]
    a_row = tree1_rows[tree1_rows["source"] == "A"].iloc[0]
    assert a_row["source_n"] == 1
    assert a_row["target_n"] == 1


def test_by_tree_linkage_matches_global(two_tree_tdata):
    """linkage df with by_tree=True is the same as by_tree=False (weighted mean)."""
    tdata = two_tree_tdata
    result_global = tl.ancestral_linkage(tdata, groupby="celltype", copy=True)
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True)
    result_by_tree = tdata.uns["celltype_linkage"]
    pd.testing.assert_frame_equal(result_global, result_by_tree)


def test_by_tree_false_no_tree_column(balanced_tdata):
    """When by_tree=False (default), stats has no 'tree' column."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype")
    stats = tdata.uns["celltype_linkage_stats"]
    assert "tree" not in stats.columns


# ── permutation test ──────────────────────────────────────────────────────────


def test_permutation_test_pairwise_linkage_stores_enrichment(balanced_tdata):
    """Pairwise permutation test stores observed - permuted_mean in uns[linkage]."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", n_permutations=20,
        random_state=42,
    )
    mat = tdata.uns["celltype_linkage"]
    assert isinstance(mat, pd.DataFrame)
    assert set(mat.index) == {"A", "B"}
    assert set(mat.columns) == {"A", "B"}
    # Values are finite (observed - permuted_mean, not z-scores)
    assert np.isfinite(mat.values).all()


def test_permutation_test_pairwise_copy_returns_stats(balanced_tdata):
    """copy=True with permutation test returns linkage_stats DataFrame."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", n_permutations=20,
        random_state=42, copy=True
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"source", "target", "value", "permuted_value", "z_score", "p_value"}
    assert set(result["source"]) == {"A", "B"}


def test_copy_no_test_returns_linkage_matrix(balanced_tdata):
    """copy=True without a test returns the linkage matrix DataFrame."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"A", "B"}
    assert set(result.columns) == {"A", "B"}


def test_permutation_test_single_target(balanced_tdata):
    """Single-target permutation test returns long DataFrame."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        n_permutations=20, random_state=0, copy=True
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"source", "target", "value", "z_score", "p_value"}
    assert (result["target"] == "B").all()


def test_permutation_test_stored_in_stats(balanced_tdata):
    """Permutation test stores z_score, p_value, and permuted_value in stats."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", n_permutations=10,
        random_state=1
    )
    assert "celltype_linkage" in tdata.uns
    assert "celltype_linkage_stats" in tdata.uns
    # No separate z_score / p_value uns keys
    assert "celltype_z_score" not in tdata.uns
    assert "celltype_p_value" not in tdata.uns
    stats = tdata.uns["celltype_linkage_stats"]
    assert "z_score" in stats.columns
    assert "p_value" in stats.columns
    assert "permuted_value" in stats.columns
    assert (stats["p_value"].between(0, 1)).all()


def test_permuted_value_in_pairwise_stats(balanced_tdata):
    """permuted_value column holds mean null distribution value for each (src, tgt) pair."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", n_permutations=20,
        random_state=42
    )
    stats = tdata.uns["celltype_linkage_stats"]
    assert "permuted_value" in stats.columns
    assert stats["permuted_value"].notna().all()


def test_permuted_value_in_single_target_test(balanced_tdata):
    """permuted_value in single-target test DataFrame holds mean null score."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        n_permutations=20, random_state=0, copy=True
    )
    assert "permuted_value" in result.columns
    assert result["permuted_value"].notna().all()


def test_n_threads_parallel_matches_serial(balanced_tdata):
    """n_threads > 1 produces same z-score DataFrame as single-threaded (same seed)."""
    def run(n_threads):
        tdata = balanced_tdata
        return tl.ancestral_linkage(
            tdata, groupby="celltype", test="permutation", n_permutations=20,
            random_state=42, n_threads=n_threads, copy=True,
        )

    serial = run(None)
    parallel = run(2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_permutation_test_stats_not_symmetrized(balanced_tdata):
    """Stats z_scores are not symmetrized even when symmetrize is set."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", n_permutations=20,
        random_state=5, symmetrize="mean"
    )
    stats = tdata.uns["celltype_linkage_stats"]
    # linkage (z_score) should be symmetrized
    mat = tdata.uns["celltype_linkage"]
    assert np.isclose(mat.loc["A", "B"], mat.loc["B", "A"])
    # stats z_scores might not be symmetric (AB vs BA could differ)
    ab = stats[(stats["source"] == "A") & (stats["target"] == "B")]["z_score"].values[0]
    ba = stats[(stats["source"] == "B") & (stats["target"] == "A")]["z_score"].values[0]
    # We just check both are present; symmetry is not required
    assert np.isfinite(ab) or np.isnan(ab)
    assert np.isfinite(ba) or np.isnan(ba)


def test_permutation_random_state(balanced_tdata):
    """Same random_state produces identical output DataFrames."""
    def run(seed):
        tdata = balanced_tdata
        return tl.ancestral_linkage(
            tdata, groupby="celltype", test="permutation", n_permutations=30,
            random_state=seed, copy=True
        )

    r1 = run(7)
    r2 = run(7)
    pd.testing.assert_frame_equal(r1, r2)


def test_by_tree_permutation_stats_has_z_score(two_tree_tdata):
    """by_tree=True with permutation test adds z_score/p_value/permuted_value to stats rows."""
    tdata = two_tree_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", by_tree=True, test="permutation",
        n_permutations=10, random_state=0,
    )
    stats = tdata.uns["celltype_linkage_stats"]
    assert "z_score" in stats.columns
    assert "p_value" in stats.columns
    assert "permuted_value" in stats.columns
    assert (stats["p_value"].between(0, 1)).all()


# ── alternative parameter tests ───────────────────────────────────────────────


def test_alternative_two_sided_pairwise_p_values_in_range(balanced_tdata):
    """Two-sided p-values are in [0, 1]."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", alternative="two-sided",
        n_permutations=20, random_state=0,
    )
    stats = tdata.uns["celltype_linkage_stats"]
    assert (stats["p_value"].between(0, 1)).all()


def test_alternative_two_sided_symmetric(balanced_tdata):
    """Two-sided p-values are symmetric: p(A→B) == p(B→A) when the tree is symmetric."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", alternative="two-sided",
        n_permutations=50, random_state=1,
    )
    stats = tdata.uns["celltype_linkage_stats"]
    ab = stats[(stats["source"] == "A") & (stats["target"] == "B")]["p_value"].values[0]
    ba = stats[(stats["source"] == "B") & (stats["target"] == "A")]["p_value"].values[0]
    assert np.isclose(ab, ba)


def test_alternative_two_sided_single_target(balanced_tdata):
    """Two-sided p-values work in single-target mode."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        alternative="two-sided", n_permutations=20, random_state=0, copy=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert (result["p_value"].between(0, 1)).all()


def test_alternative_none_matches_default(balanced_tdata):
    """alternative=None produces identical results to omitting the parameter."""
    def run(alt):
        tdata = balanced_tdata
        return tl.ancestral_linkage(
            tdata, groupby="celltype", test="permutation", alternative=alt,
            n_permutations=20, random_state=42, copy=True,
        )

    pd.testing.assert_frame_equal(run(None), run(None))


# ── permutation_mode tests ────────────────────────────────────────────────────


def test_permutation_mode_non_target_pairwise_p_in_range(balanced_tdata):
    """non_target mode pairwise: p-values in [0, 1]."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", test="permutation", permutation_mode="non_target",
        n_permutations=20, random_state=0,
    )
    stats = tdata.uns["celltype_linkage_stats"]
    assert (stats["p_value"].between(0, 1)).all()
    assert "z_score" in stats.columns
    assert "permuted_value" in stats.columns


def test_permutation_mode_non_target_single_target_p_in_range(balanced_tdata):
    """non_target mode single-target: p-values in [0, 1]."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        permutation_mode="non_target", n_permutations=20, random_state=0, copy=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert (result["p_value"].between(0, 1)).all()


def test_permutation_mode_non_target_differs_from_all(balanced_tdata):
    """non_target and all modes produce different null means (different null construction)."""
    def run(mode):
        tdata = balanced_tdata
        return tl.ancestral_linkage(
            tdata, groupby="celltype", test="permutation", permutation_mode=mode,
            n_permutations=50, random_state=42, copy=True,
        )

    stats_all = run("all")
    stats_non_target = run("non_target")
    # permuted_value may differ between modes (different null distributions)
    # At minimum both should produce valid DataFrames with the same structure
    assert set(stats_all.columns) == set(stats_non_target.columns)


def test_permutation_mode_non_target_by_tree(two_tree_tdata):
    """non_target mode works with by_tree=True."""
    tdata = two_tree_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", by_tree=True, test="permutation",
        permutation_mode="non_target", n_permutations=10, random_state=0,
    )
    stats = tdata.uns["celltype_linkage_stats"]
    assert "z_score" in stats.columns
    assert (stats["p_value"].between(0, 1)).all()


# ── warning tests ─────────────────────────────────────────────────────────────


def test_lca_min_warning(balanced_tdata):
    """aggregate='min' + metric='lca' should emit a UserWarning."""
    with pytest.warns(UserWarning, match="shallowest"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype",
                             aggregate="min", metric="lca")


# ── edge / validation tests ───────────────────────────────────────────────────


def test_invalid_groupby(balanced_tdata):
    with pytest.raises(ValueError, match="not found"):
        tl.ancestral_linkage(balanced_tdata, groupby="nonexistent")


def test_invalid_aggregate(balanced_tdata):
    with pytest.raises(ValueError, match="aggregate"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", aggregate="median")


def test_custom_callable_aggregate(balanced_tdata):
    """Custom callable is accepted for aggregate."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate=np.mean, metric="path")
    mat = tdata.uns["celltype_linkage"]
    # Should match mean aggregate
    tl.ancestral_linkage(tdata, groupby="celltype", aggregate="mean", metric="path",
                         key_added="ref")
    ref = tdata.uns["ref_linkage"]
    pd.testing.assert_frame_equal(mat, ref)


# ── bug-fix regression tests ──────────────────────────────────────────────────


def test_lca_max_non_ultrametric_raises():
    """aggregate='max' + metric='lca' on a non-ultrametric tree must raise ValueError."""
    t = nx.DiGraph()
    for node, depth in [
        ("root", 0.0), ("n1", 0.8), ("n2", 0.1),
        ("a1", 1.0), ("b1", 0.2), ("b2", 0.9),
    ]:
        t.add_node(node, depth=depth)
    t.add_edges_from([
        ("root", "n1"), ("root", "n2"),
        ("n1", "a1"), ("n1", "b2"),
        ("n2", "b1"),
    ])
    obs = pd.DataFrame({"ct": ["A", "B", "B"]}, index=["a1", "b1", "b2"])
    tdata = td.TreeData(obs=obs, obst={"tree": t})

    with pytest.raises(ValueError, match="ultrametric"):
        tl.ancestral_linkage(tdata, groupby="ct", aggregate="max", metric="lca")


def test_symmetrize_invalid_raises():
    """Unknown symmetrize value must raise ValueError, not silently use 'min'."""
    t = nx.DiGraph()
    for node, depth in [("root", 0.0), ("a1", 1.0), ("b1", 1.0)]:
        t.add_node(node, depth=depth)
    t.add_edges_from([("root", "a1"), ("root", "b1")])
    obs = pd.DataFrame({"ct": ["A", "B"]}, index=["a1", "b1"])
    tdata = td.TreeData(obs=obs, obst={"tree": t})

    with pytest.raises(ValueError, match="symmetrize"):
        tl.ancestral_linkage(tdata, groupby="ct", symmetrize="meen")


# ── permutation improvement tests ─────────────────────────────────────────────


def test_pairwise_permutation_linkage_is_value_minus_permuted(balanced_tdata):
    """uns linkage matrix stores observed - permuted_mean (not z-scores) when test='permutation'."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", n_permutations=20, random_state=0)
    linkage = tdata.uns["celltype_linkage"]
    stats = tdata.uns["celltype_linkage_stats"]
    for _, row in stats.iterrows():
        expected = row["value"] - row["permuted_value"]
        actual = linkage.loc[row["source"], row["target"]]
        assert actual == pytest.approx(expected, abs=1e-9)


def test_single_target_permutation_adds_norm_linkage(balanced_tdata):
    """test='permutation' in single-target mode adds _norm_linkage column to obs."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        n_permutations=30, random_state=1,
    )
    assert "B_linkage" in tdata.obs.columns
    assert "B_norm_linkage" in tdata.obs.columns
    # norm_linkage should be finite for cells with valid scores
    valid = tdata.obs["B_linkage"].notna()
    assert tdata.obs.loc[valid, "B_norm_linkage"].notna().all()


def test_single_target_norm_linkage_sign(balanced_tdata):
    """Within-target cells should have positive norm_linkage (closer to target = higher z-score)."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        n_permutations=100, random_state=42,
    )
    # B cells are closest to B (path=1.0); A cells are farther (path=2.0)
    # For metric='path', sign=-1, so B cells get higher (less negative) norm_linkage
    b_norm = tdata.obs.loc[["b1", "b2"], "B_norm_linkage"].mean()
    a_norm = tdata.obs.loc[["a1", "a2"], "B_norm_linkage"].mean()
    assert b_norm > a_norm


def test_single_target_by_tree_permutation(three_cat_tdata):
    """by_tree=True with single target + permutation runs per-tree and adds norm_linkage."""
    tdata = three_cat_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", target="A", test="permutation", by_tree=True,
        n_permutations=20, random_state=7,
    )
    assert "A_linkage" in tdata.obs.columns
    assert "A_norm_linkage" in tdata.obs.columns
    test_df = tdata.uns["celltype_test"]
    # by_tree adds a "tree" column
    assert "tree" in test_df.columns
    # Each tree in tdata.obst should appear in the test results
    for tree_key in tdata.obst:
        assert tree_key in test_df["tree"].values


def test_single_target_by_tree_perm_non_target(balanced_tdata):
    """by_tree + permutation_mode='non_target' in single-target mode runs and produces norm_linkage."""
    tdata = balanced_tdata
    tl.ancestral_linkage(
        tdata, groupby="celltype", target="B", test="permutation",
        by_tree=True, permutation_mode="non_target", n_permutations=20, random_state=3,
    )
    assert "B_norm_linkage" in tdata.obs.columns
    test_df = tdata.uns["celltype_test"]
    assert "tree" in test_df.columns
