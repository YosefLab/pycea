from collections.abc import Callable
from typing import Literal

import numpy as np
import treedata as td

_MetricFn = Callable[[np.ndarray, np.ndarray], float]

_Metric = Literal[
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "cosine",
    "correlation",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "kulsinski",
    "l1",
    "l2",
    "mahalanobis",
    "minkowski",
    "manhattan",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


def _lca_distance(tree, depth_key, node1, node2, lca):
    """Compute the lca distance between two nodes in a tree."""
    if node1 == node2:
        return tree.nodes[node1][depth_key]
    else:
        return tree.nodes[lca][depth_key]


def _path_distance(tree, depth_key, node1, node2, lca):
    """Compute the path distance between two nodes in a tree."""
    if node1 == node2:
        return 0
    else:
        return abs(tree.nodes[node1][depth_key] + tree.nodes[node2][depth_key] - 2 * tree.nodes[lca][depth_key])


_TreeMetricFn = Callable[[td.TreeData, str, str, str, str], float]

_TreeMetric = Literal["lca", "path"]


def _get_tree_metric(metric: str) -> _TreeMetricFn:
    if metric == "lca":
        return _lca_distance
    elif metric == "path":
        return _path_distance
    else:
        raise ValueError(f"Unknown metric: {metric}. Valid metrics are 'lca' and 'path'.")
