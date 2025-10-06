import numpy as np
import pandas as pd
import pytest
import scipy as sp
import treedata as td

from pycea.tl.autocorr import autocorr


@pytest.fixture
def tdata():
    tdata = td.TreeData(
        obs=pd.DataFrame({"value": [1, 1, 0]}, index=["A", "B", "C"]),
        var=pd.DataFrame(index=["1", "2", "3"]),
        X=np.array([[1, 0, 1], [1, 1, 2], [0, 1, 5]]),
        obsp={
            "connectivities": sp.sparse.csr_matrix(([1, 1, 1, 1, 1], ([0, 0, 1, 1, 2], [0, 1, 0, 1, 2])), shape=(3, 3))
        },
    )
    tdata.obsm["values"] = pd.DataFrame(index=["A", "B", "C"], columns=["X", "Y"], data=[[0, 0], [1, 1], [2, 2]])
    yield tdata


def test_moran(tdata):
    autocorr(tdata, keys="value", connect_key="connectivities", method="moran")
    assert "moranI" in tdata.uns.keys()
    assert tdata.uns["moranI"]["autocorr"].values[0] == pytest.approx(0.8)
    result = autocorr(tdata, keys=["1", "2"], connect_key="connectivities", method="moran", copy=True)
    assert result is not None
    assert tdata.uns["moranI"].shape == (2, 3)
    assert list(result.index.values) == ["1", "2"]
    assert result.autocorr.values == pytest.approx([0.8, 0.2])
    assert result.pval_norm.values == pytest.approx([1.84e-12, 9.14e-5], rel=1e-2)
    result = autocorr(tdata, connect_key="connectivities", method="moran", copy=True, keys="values")
    assert result is not None
    assert list(result.index.values) == ["X", "Y"]
    result = autocorr(tdata, keys="1", connect_key="connectivities", method="moran", copy=True)
    assert result is not None
    assert list(result.index.values) == ["1"]
    assert result.autocorr.values == pytest.approx([0.8])
    assert result.pval_norm.values == pytest.approx([1.84e-12], rel=1e-2)


def test_geary(tdata):
    autocorr(tdata, keys="value", connect_key="connectivities", method="geary")
    assert "gearyC" in tdata.uns.keys()
    assert tdata.uns["gearyC"]["autocorr"].values[0] == pytest.approx(0)
    result = autocorr(tdata, connect_key="connectivities", method="geary", copy=True)
    print(result)
    assert tdata.uns["gearyC"].shape == (3, 3)
    assert result is not None
    assert list(result.index.values) == ["1", "2", "3"]
    assert result.autocorr.values == pytest.approx([0.0, 0.6, 0.04615], rel=1e-2)
    assert result.pval_norm.values == pytest.approx([4.51e-8, 1.62e-2, 1.71e-7], rel=1e-2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
