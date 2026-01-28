import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--internet-tests", action="store_true", default=False, help="run tests that require internet access"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "internet: mark test as requiring internet access")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--internet-tests"):
        return
    skip_internet = pytest.mark.skip(reason="need --internet-tests option to run")
    for item in items:
        if "internet" in item.keywords:
            item.add_marker(skip_internet)
