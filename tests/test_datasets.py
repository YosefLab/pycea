import pytest

from pycea.datasets import colgan26, koblan25, packer19, yang22, yu26


@pytest.mark.internet
def test_packer19():
    tdata = packer19()
    assert tdata.shape == (988, 20222)
    assert set(tdata.obst.keys()) == {"tree"}


@pytest.mark.internet
def test_yang22():
    tdata = yang22(tumors="3435_NT_T1")
    assert tdata.shape == (1109, 2000)
    assert set(tdata.obst.keys()) == {"3435_NT_T1"}


@pytest.mark.internet
def test_koblan25():
    tdata = koblan25(experiment="tumor")
    assert tdata.shape == (145954, 175)
    assert set(tdata.obst.keys()) == {"tree"}
    tdata = koblan25(experiment="barcoding")
    assert tdata.shape == (3108, 2000)
    assert set(tdata.obst.keys()) == {"tree"}


@pytest.mark.slow
@pytest.mark.internet
def test_colgan26():
    tdata = colgan26(embryos="E7.5-R1")
    assert set(tdata.obs["embryo"].unique()) == {"E7.5-R1"}
    assert set(tdata.obst.keys()) == {"E7.5-R1-C1", "E7.5-R1-C2"}
    assert "X_umap" in tdata.obsm


@pytest.mark.slow
@pytest.mark.internet
def test_yu26():
    tdata = yu26()
    assert tdata.shape == (640012, 0)
    assert set(tdata.obst.keys()) == {"tree"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])
