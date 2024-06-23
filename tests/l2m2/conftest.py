import warnings


def pytest_runtest_setup(item):
    # Ignore never awaited warning in test_llm_client
    if "test_llm_client.py" in item.nodeid:
        warnings.filterwarnings("ignore", category=RuntimeWarning)


def pytest_runtest_teardown(item):
    if "test_llm_client.py" in item.nodeid:
        warnings.resetwarnings()
