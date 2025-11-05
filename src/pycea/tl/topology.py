import networkx as nx
import treedata as td
from scipy.special import comb as nCk

from pycea.utils import (
    get_root,
    get_trees,
)


def compute_expansion_pvalues(
    tdata: td.TreeData,
    tree: str | None = None,
    min_clade_size: int = 10,
    min_depth: int = 1,
    copy: bool = False,
) -> td.TreeData | None:
    """Compute expansion p-values on a tree.

    Uses the methodology described in Yang, Jones et al, BioRxiv (2021) to
    assess the expansion probability of a given subclade of a phylogeny.
    Mathematical treatment of the coalescent probability is described in
    Griffiths and Tavare, Stochastic Models (1998).

    The probability computed corresponds to the probability that, under a simple
    neutral coalescent model, a given subclade contains the observed number of
    cells; in other words, a one-sided p-value. Often, if the probability is
    less than some threshold (e.g., 0.05), this might indicate that there exists
    some subclade under this node to which this expansion probability can be
    attributed (i.e. the null hypothesis that the subclade is undergoing
    neutral drift can be rejected).

    This function will add an attribute "expansion_pvalue" to tree nodes.

    On a typical balanced tree, this function performs in O(n log n) time,
    but can be up to O(n^3) on highly unbalanced trees. A future implementation
    may optimize this to O(n) time.

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
        The `obst` key of the tree to use. If `None` and only one tree is
        present, that tree is used. If `None` and multiple trees are present,
        raises an error.
    copy
        If True, return a copy of the TreeData with attributes added.
        If False, modify in place and return None. Default is False.

    Returns
    -------
    If `copy=False`, returns `None` (tree modified in place).
    If `copy=True`, returns a new :class:`TreeData` object with expansion
    p-values added to tree nodes.
    """
    if copy:
        tdata = tdata.copy()
    trees = get_trees(tdata, tree)
    if len(trees) != 1:
        raise ValueError(
            f"Expected exactly one tree, but found {len(trees)}. "
            "Please specify which tree to use with the 'tree' parameter."
        )
    tree_key, t = next(iter(trees.items()))
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

    nx.set_node_attributes(t, 1.0, "expansion_pvalue")

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
            t.nodes[child]["expansion_pvalue"] = float(p)

    return tdata if copy else None
