import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless -m integration is passed."""
    if "integration" in config.getoption("-m", default=""):
        return
    skip_integration = pytest.mark.skip(reason="pass -m integration to run")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
