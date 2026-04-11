"""Pytest configuration and shared fixtures."""

import os
import signal
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
    config.addinivalue_line(
        "markers",
        "ddp: Multi-GPU DDP tests (requires srun with >=2 GPUs)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on markers."""
    skip_v1 = pytest.mark.skip(reason="v1: legacy test needs updating")
    skip_gpu = pytest.mark.skip(reason="no GPU available")
    skip_ddp = pytest.mark.skip(reason="DDP requires >=2 GPUs (use srun --gpus=N)")
    for item in items:
        if "v1" in item.keywords:
            item.add_marker(skip_v1)
        elif "ddp" in item.keywords and torch.cuda.device_count() < 2:
            item.add_marker(skip_ddp)
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
    from stable_pretraining.registry import _db as _registry_db
    from stable_pretraining.registry import logger as _registry_logger

    cfg = get_config()
    original = cfg._cache_dir
    cfg._cache_dir = str(tmp_path)

    # Wrap RegistryDB so that any path requested auto-starts a server first.
    # This handles tests that change cache_dir after this fixture runs
    # (e.g. the cache_dir fixture in test_cache_dir.py).
    _original_RegistryDB = _registry_db.RegistryDB
    _started_db_paths = []

    def _auto_ensure_RegistryDB(db_path):  # noqa: N802
        _registry_db.ensure_server(db_path)
        _started_db_paths.append(db_path)
        return _original_RegistryDB(db_path)

    _registry_db.RegistryDB = _auto_ensure_RegistryDB
    _registry_logger.RegistryDB = _auto_ensure_RegistryDB

    yield

    # Teardown: restore original RegistryDB
    _registry_db.RegistryDB = _original_RegistryDB
    _registry_logger.RegistryDB = _original_RegistryDB

    # Kill any servers started during this test (never touches user's registry)
    for db_path in _started_db_paths:
        info = _registry_db._read_discovery(db_path)
        if info and info.get("pid"):
            try:
                os.kill(info["pid"], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    cfg._cache_dir = original
    # Clean up empty outputs/ dir that Hydra creates when cache_dir is None
    outputs = Path("outputs")
    if outputs.is_dir():
        shutil.rmtree(outputs, ignore_errors=True)
