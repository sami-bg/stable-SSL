"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no GPU required)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (slow, may require GPU)"
    )
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line(
        "markers", "download: Tests that download data from the internet"
    )
    config.addinivalue_line(
        "markers", "v1: Legacy tests that need updating (auto-skipped)"
    )
    config.addinivalue_line(
        "markers",
        "regression: Regression tests (all methods, fake data, CPU-only, checks registry)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on markers."""
    skip_v1 = pytest.mark.skip(reason="v1: legacy test needs updating")
    skip_gpu = pytest.mark.skip(reason="no GPU available")
    for item in items:
        if "v1" in item.keywords:
            item.add_marker(skip_v1)
        elif "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


@pytest.fixture
def device():
    """Fixture to get appropriate device for tests."""
    if torch.cuda.is_available() and not os.environ.get("FORCE_CPU"):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_batch(device):
    """Create a mock batch of data."""
    batch_size = 4
    return {
        "image": torch.randn(batch_size, 3, 224, 224, device=device),
        "label": torch.randint(0, 10, (batch_size,), device=device),
        "index": torch.arange(batch_size, device=device),
    }


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")


@pytest.fixture(autouse=True)
def _isolate_spt_config(tmp_path):
    """Point spt cache_dir to a temp folder for every test.

    This ensures no test writes files (environment.json, registry.db,
    CSV logs, etc.) to the working directory.  pytest auto-cleans
    tmp_path after each test.

    Some tests intentionally reset cache_dir to ``None`` (e.g. to test
    the legacy code path).  When that happens, Hydra/Lightning may
    create an empty ``outputs/`` directory in CWD.  We clean it up in
    teardown so it never persists.
    """
    import shutil

    from stable_pretraining._config import get_config

    cfg = get_config()
    original = cfg._cache_dir
    cfg._cache_dir = str(tmp_path)
    yield
    cfg._cache_dir = original
    # Clean up empty outputs/ dir that Hydra creates when cache_dir is None
    outputs = Path("outputs")
    if outputs.is_dir():
        shutil.rmtree(outputs, ignore_errors=True)
