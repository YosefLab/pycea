import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from pycea.tl import ancestral_linkage, clades, fitness, n_extant, tree_distance, tree_neighbors
from pycea.utils import get_depth_key


@pytest.fixture
def tdata():
    """Tree with only a 'time' attribute and uns['default_depth'] pointing at it."""
    tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, {n: f"n{n}" for n in tree.nodes})
    levels = nx.single_source_shortest_path_length(tree, "n0")
    nx.set_node_attributes(tree, {n: 0.5 * (d + 1) ** 1.5 for n, d in levels.items()}, "time")
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    obs = pd.DataFrame({"group": ["a", "b"] * 4}, index=leaves)
    tdata = td.TreeData(obs=obs, obst={"tree": tree})
    tdata.uns["default_depth"] = "time"
    return tdata


def test_get_depth_key(tdata):
    assert get_depth_key(tdata) == "time"
    assert get_depth_key(tdata, "depth") == "depth"
    del tdata.uns["default_depth"]
    assert get_depth_key(tdata) == "depth"


def _same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif hasattr(a, "toarray"):
        assert np.allclose(a.toarray(), b.toarray())
    else:
        assert np.allclose(a, b)


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (tree_distance, {"metric": "path", "copy": True}),
        (tree_distance, {"metric": "lca", "copy": True}),
        (tree_neighbors, {"n_neighbors": 2, "metric": "path", "random_state": 0, "copy": True}),
        (tree_neighbors, {"n_neighbors": 2, "metric": "lca", "random_state": 0, "copy": True}),
        (clades, {"depth": 3.0, "copy": True}),
        (n_extant, {"bins": 3, "copy": True}),
        (ancestral_linkage, {"groupby": "group", "metric": "lca", "aggregate": "max", "min_size": 1, "copy": True}),
        (fitness, {"method": "lbi", "random_state": 0, "copy": True}),
    ],
)
def test_default_depth_matches_explicit(tdata, func, kwargs):
    # Without default_depth the "depth" attribute is missing, so the default would fail
    default = func(tdata, copy=True, **{k: v for k, v in kwargs.items() if k != "copy"})
    explicit = func(tdata.copy(), depth_key="time", **kwargs)
    if isinstance(default, tuple):
        for a, b in zip(default, explicit, strict=True):
            _same(a, b)
    else:
        _same(default, explicit)
    del tdata.uns["default_depth"]
    with pytest.raises(ValueError, match="depth"):
        func(tdata, **kwargs)
