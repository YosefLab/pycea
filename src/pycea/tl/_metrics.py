from collections.abc import Callable
from typing import Literal, Union

import numpy as np

_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
_MetricScipySpatial = Literal[
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]
