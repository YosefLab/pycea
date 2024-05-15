import pytest
import treedata as td

_tdata = td.read_h5ad("tests/data/tdata.h5ad")


@pytest.fixture(scope="session")
def tdata() -> td.TreeData:
    return _tdata
