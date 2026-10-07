import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--internet-tests", action="store_true", default=False, help="run tests that require internet access"
    )
    parser.addoption("--slow-tests", action="store_true", default=False, help="run slow tests (e.g. large datasets)")


def pytest_configure(config):
    config.addinivalue_line("markers", "internet: mark test as requiring internet access")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    skip_internet = pytest.mark.skip(reason="need --internet-tests option to run")
    skip_slow = pytest.mark.skip(reason="need --slow-tests option to run")
    for item in items:
        if "internet" in item.keywords and not config.getoption("--internet-tests"):
            item.add_marker(skip_internet)
        if "slow" in item.keywords and not config.getoption("--slow-tests"):
            item.add_marker(skip_slow)
