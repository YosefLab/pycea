from collections.abc import Sequence
from typing import Literal, overload

import networkx as nx
import pandas as pd
import treedata as td
from scipy.special import comb as nCk

from pycea.utils import get_keyed_node_data, get_root, get_trees


@overload
def expansion_test(
    tdata: td.TreeData,
    tree: str | Sequence[str] | None = None,
    min_clade_size: int = 10,
    min_depth: int = 1,
    key_added: str = "expansion_pvalue",
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...
@overload
def expansion_test(
    tdata: td.TreeData,
    tree: str | Sequence[str] | None = None,
    min_clade_size: int = 10,
    min_depth: int = 1,
    key_added: str = "expansion_pvalue",
    copy: Literal[True, False] = False,
) -> None: ...
def expansion_test(
    tdata: td.TreeData,
    tree: str | Sequence[str] | None = None,
    min_clade_size: int = 10,
    min_depth: int = 1,
    key_added: str = "expansion_pvalue",
    copy: Literal[True, False] = False,
) -> pd.DataFrame | None:
    """Compute expansion p-values on a tree.

    Uses the methodology described in :cite:`Yang_2022` to
    assess the expansion probability of a given subclade of a phylogeny.
    Mathematical treatment of the coalescent probability is described in :cite:`Griffiths_1998`.

    The probability computed corresponds to the probability that, under a simple
    neutral coalescent model, a given subclade contains the observed number of
    cells; in other words, a one-sided p-value. Often, if the probability is
    less than some threshold (e.g., 0.05), this might indicate that there exists
    some subclade under this node to which this expansion probability can be
    attributed (i.e. the null hypothesis that the subclade is undergoing
    neutral drift can be rejected).

    This function will add an attribute to tree nodes storing the expansion p-value.

    On a typical balanced tree, this function performs in O(n) time.

    Parameters
    ----------
    tdata
        TreeData object containing a phylogenetic tree.
    min_clade_size
        Minimum number of leaves in a subtree to be considered. Default is 10.
    min_depth
        Minimum depth of clade to be considered. Depth is measured in number
        of nodes from the root, not branch lengths. Default is 1.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    key_added
        Attribute key where expansion p-values will be stored in tree nodes.
        Default is "expansion_pvalue".
    copy
        If True, return a :class:`pandas.DataFrame` with attributes added.
        If False, modify in place and return None. Default is False.

    Returns
    -------
    Returns `None` if ``copy=False``, otherwise returns a :class:`pandas.DataFrame` with expansion pvalues.

    Sets the following fields:

    * tdata.obst[tree].nodes[key_added] : `float`
        - Expansion pvalue for each node.
    """
    trees = get_trees(tdata, tree)

    for _tree_key, t in trees.items():
        root = get_root(t)
        # instantiate attributes
        leaf_counts = {}
        for node in nx.dfs_postorder_nodes(t, root):
            if t.out_degree(node) == 0:
                leaf_counts[node] = 1
            else:
                leaf_counts[node] = sum(leaf_counts[child] for child in t.successors(node))

        depths = {root: 0}
        for u, v in nx.dfs_edges(t, root):
            depths[v] = depths[u] + 1

        nx.set_node_attributes(t, 1.0, key_added)

        for node in t.nodes():
            n = leaf_counts[node]
            children = list(t.successors(node))
            k = len(children)

            if k == 0:
                continue

            for child in children:
                b = leaf_counts[child]
                depth = depths[child]

                # Apply filters
                if b < min_clade_size:
                    continue
                if depth < min_depth:
                    continue

                p = nCk(n - b, k - 1) / nCk(n - 1, k - 1)
                t.nodes[child][key_added] = float(p)

    if copy:
        df = get_keyed_node_data(tdata, keys=key_added, tree=tree, slot="obst")
        if len(trees) == 1:
            df.index = df.index.droplevel(0)
        return df
    else:
        return None
