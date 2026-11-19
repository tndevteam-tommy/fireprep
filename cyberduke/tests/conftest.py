import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="run smoke tests that boot the real game (needs a display / Xvfb)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--smoke"):
        return
    skip = pytest.mark.skip(reason="smoke tests only run with --smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: boots the real game window")
