from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import pandas as pd
import treedata as td

from pycea.utils import get_keyed_edge_data, get_keyed_node_data, get_trees


def _infer_edge_keys(trees: Mapping[str, nx.DiGraph]) -> list[str]:
    keyset: set[str] = set()
    for t in trees.values():
        for _, _, attrs in t.edges(data=True):
            keyset.update(attrs.keys())
    return sorted(keyset)


def _infer_node_keys(trees: Mapping[str, nx.DiGraph]) -> list[str]:
    keyset: set[str] = set()
    for t in trees.values():
        for _, attrs in t.nodes(data=True):
            keyset.update(attrs.keys())
    return sorted(keyset)


def _maybe_drop_tree_index(df: pd.DataFrame, n_trees: int) -> pd.DataFrame:
    if n_trees == 1 and "tree" in df.index.names:
        return df.droplevel("tree")
    return df


def edge_df(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Return edge-level metadata as a :class:`~pandas.DataFrame`.

    When multiple trees are selected, the returned DataFrame has a
    ``("tree", "edge")`` index. Otherwise the index is simply ``"edge"``.
    """
    trees = get_trees(tdata, tree)
    keys = _infer_edge_keys(trees)
    df = get_keyed_edge_data(tdata, keys, tree=list(trees.keys()), slot="obst")
    return _maybe_drop_tree_index(df, len(trees))


def node_df(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Return node-level metadata as a :class:`~pandas.DataFrame`.

    When multiple trees are selected, the returned DataFrame has a
    ``("tree", "node")`` index. Otherwise the index is simply ``"node"``.
    """
    trees = get_trees(tdata, tree)
    keys = _infer_node_keys(trees)
    df = get_keyed_node_data(tdata, keys, tree=list(trees.keys()), slot="obst")
    return _maybe_drop_tree_index(df, len(trees))
