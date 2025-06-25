import pytest

from pycea.datasets import koblan25, packer19, yang22


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
