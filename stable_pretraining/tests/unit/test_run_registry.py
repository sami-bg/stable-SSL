"""Unit tests for the SQLite run registry (db, logger, query)."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from stable_pretraining.registry import _db as _registry_db
from stable_pretraining.registry.logger import RegistryLogger, _flatten_params
from stable_pretraining.registry.query import Registry, RunRecord, open_registry

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a fresh temporary database."""
    return str(tmp_path / "test_registry.db")


@pytest.fixture
def db(tmp_db):
    """Return an open RegistryDB instance."""
    _db = _registry_db.RegistryDB(tmp_db)
    yield _db
    _db.close()


@pytest.fixture
def populated_db(db):
    """Insert a few sample runs and return the DB."""
    db.insert_run(
        "run-001",
        status="completed",
        run_dir="/tmp/runs/run-001",
        config={"trainer.max_epochs": 100, "module.lr": 0.01},
        hparams={"lr": 0.01, "epochs": 100},
        tags=["resnet", "baseline", "sweep:100"],
        notes="First baseline run",
    )
    db.insert_run(
        "run-002",
        status="completed",
        run_dir="/tmp/runs/run-002",
        config={"trainer.max_epochs": 100, "module.lr": 0.1},
        hparams={"lr": 0.1, "epochs": 100},
        tags=["resnet", "high-lr", "sweep:100"],
    )
    db.insert_run(
        "run-003",
        status="running",
        run_dir="/tmp/runs/run-003",
        config={"trainer.max_epochs": 50},
        hparams={"lr": 0.001, "epochs": 50},
        tags=["vit", "sweep:200"],
    )
    # Add summary to run-001 and run-002
    db.update_run("run-001", summary={"val_acc": 0.85, "train_loss": 0.12})
    db.update_run("run-002", summary={"val_acc": 0.92, "train_loss": 0.08})
    return db


# ============================================================================
# RegistryDB
# ============================================================================


class TestRegistryDB:
    """Tests for the RegistryDB SQLite wrapper."""

    def test_insert_and_get(self, db):
        db.insert_run(
            "test-run",
            run_dir="/tmp/test",
            config={"lr": 0.01},
            hparams={"lr": 0.01},
        )
        run = db.get_run("test-run")
        assert run is not None
        assert run["run_id"] == "test-run"
        assert run["status"] == "running"
        assert run["config"] == {"lr": 0.01}
        assert run["hparams"] == {"lr": 0.01}
        assert run["summary"] == {}
        assert run["tags"] == []
        assert run["notes"] == ""

    def test_get_nonexistent(self, db):
        assert db.get_run("nonexistent") is None

    def test_update_run(self, db):
        db.insert_run("test-run", run_dir="/tmp/test")
        db.update_run("test-run", status="completed", checkpoint_path="/tmp/ckpt.pt")
        run = db.get_run("test-run")
        assert run["status"] == "completed"
        assert run["checkpoint_path"] == "/tmp/ckpt.pt"

    def test_update_summary_json(self, db):
        db.insert_run("test-run")
        db.update_run("test-run", summary={"val_acc": 0.9, "loss": 0.1})
        run = db.get_run("test-run")
        assert run["summary"] == {"val_acc": 0.9, "loss": 0.1}

    def test_upsert_updates_fields(self, db):
        db.insert_run("test-run", status="running", config={"v": 1})
        db.insert_run("test-run", status="completed", config={"v": 2})
        run = db.get_run("test-run")
        assert run["status"] == "completed"
        assert run["config"] == {"v": 2}

    def test_requeue_preserves_created_at(self, db):
        """On requeue (same run_id re-inserted), created_at must not change."""
        db.insert_run("requeue-run", status="running", config={"epoch": 0})
        original = db.get_run("requeue-run")
        original_created = original["created_at"]

        time.sleep(0.05)  # ensure wall-clock advances

        db.insert_run("requeue-run", status="running", config={"epoch": 50})
        requeued = db.get_run("requeue-run")
        assert requeued["created_at"] == original_created
        assert requeued["updated_at"] > original_created
        assert requeued["config"] == {"epoch": 50}

    def test_query_by_status(self, populated_db):
        runs = populated_db.query_runs(status="running")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-003"

    def test_query_by_tag(self, populated_db):
        runs = populated_db.query_runs(tag="resnet")
        assert len(runs) == 2
        assert {r["run_id"] for r in runs} == {"run-001", "run-002"}

    def test_query_by_sweep_tag(self, populated_db):
        runs = populated_db.query_runs(tag="sweep:100")
        assert len(runs) == 2

    def test_query_sort_by_column(self, populated_db):
        runs = populated_db.query_runs(tag="sweep:100", sort_by="created_at")
        assert len(runs) == 2

    def test_query_sort_by_summary(self, populated_db):
        runs = populated_db.query_runs(
            tag="sweep:100", sort_by="summary.val_acc", descending=True
        )
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-002"  # 0.92 > 0.85

    def test_query_limit(self, populated_db):
        runs = populated_db.query_runs(limit=1)
        assert len(runs) == 1

    def test_query_all(self, populated_db):
        runs = populated_db.query_runs()
        assert len(runs) == 3

    def test_wal_mode_enabled(self, tmp_db):
        """WAL mode is a property of the SQLite backend, test it directly."""
        from stable_pretraining.registry._local import LocalRegistryDB

        local_db = LocalRegistryDB(tmp_db)
        conn = local_db._get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        local_db.close()

    def test_close_and_reopen(self, tmp_db):
        db = _registry_db.RegistryDB(tmp_db)
        db.insert_run("persistent-run", config={"x": 1})
        db.close()

        db2 = _registry_db.RegistryDB(tmp_db)
        run = db2.get_run("persistent-run")
        assert run is not None
        assert run["config"] == {"x": 1}
        db2.close()


# ============================================================================
# RegistryLogger
# ============================================================================


class TestRegistryLogger:
    """Tests for the RegistryLogger Lightning logger."""

    def test_name_and_version(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger._run_id = "run-42"
        assert logger.name == "registry"
        assert logger.version == "run-42"

    def test_log_hyperparams(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db, tags=["resnet"])
        logger._run_id = "run-hp"
        logger._run_dir = "/tmp/runs/run-hp"
        logger.log_hyperparams({"lr": 0.01, "batch_size": 32})

        run = logger._db.get_run("run-hp")
        assert run is not None
        assert run["hparams"]["lr"] == 0.01
        assert run["run_dir"] == "/tmp/runs/run-hp"
        assert run["status"] == "running"
        assert "resnet" in run["tags"]

    def test_log_metrics_accumulates_summary(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db, flush_every=100)
        logger._run_id = "run-metrics"

        logger.log_metrics({"train_loss": 1.0}, step=0)
        logger.log_metrics({"train_loss": 0.5, "val_acc": 0.8}, step=1)

        assert logger._summary == {"train_loss": 0.5, "val_acc": 0.8}

    def test_log_metrics_periodic_flush(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db, flush_every=2)
        logger._run_id = "run-flush"
        logger._run_dir = "/tmp/runs/run-flush"

        logger.log_hyperparams({"lr": 0.01})

        logger.log_metrics({"loss": 1.0}, step=0)
        run = logger._db.get_run("run-flush")
        assert run["summary"] == {}

        logger.log_metrics({"loss": 0.5}, step=1)
        run = logger._db.get_run("run-flush")
        assert run["summary"]["loss"] == 0.5

    def test_finalize_success(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger._run_id = "run-final"
        logger._run_dir = "/tmp/runs/run-final"

        logger.log_hyperparams({"lr": 0.01})
        logger.log_metrics({"val_acc": 0.9}, step=99)
        logger.finalize("success")

        db = _registry_db.RegistryDB(tmp_db)
        run = db.get_run("run-final")
        assert run["status"] == "completed"
        assert run["summary"]["val_acc"] == 0.9
        db.close()

    def test_finalize_failed(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger._run_id = "run-fail"
        logger._run_dir = "/tmp/runs/run-fail"

        logger.log_hyperparams({"lr": 0.01})
        logger.finalize("failed")

        db = _registry_db.RegistryDB(tmp_db)
        run = db.get_run("run-fail")
        assert run["status"] == "failed"
        db.close()

    def test_finalize_without_log_hyperparams(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger._run_id = "run-no-hp"
        logger._run_dir = "/tmp/runs/run-no-hp"

        logger.log_metrics({"loss": 0.5}, step=0)
        logger.finalize("success")

        db = _registry_db.RegistryDB(tmp_db)
        run = db.get_run("run-no-hp")
        assert run is not None
        assert run["status"] == "completed"
        db.close()

    def test_after_save_checkpoint(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger._run_id = "run-ckpt"
        logger._run_dir = "/tmp/runs/run-ckpt"
        logger.log_hyperparams({})

        mock_cb = MagicMock()
        mock_cb.best_model_path = "/tmp/runs/run-ckpt/checkpoints/best.ckpt"
        logger.after_save_checkpoint(mock_cb)

        run = logger._db.get_run("run-ckpt")
        assert run["checkpoint_path"] == "/tmp/runs/run-ckpt/checkpoints/best.ckpt"

    def test_no_run_id_is_noop(self, tmp_db):
        logger = RegistryLogger(db_path=tmp_db)
        logger.log_hyperparams({"lr": 0.01})
        logger.log_metrics({"loss": 0.5}, step=0)
        logger.finalize("success")

    def test_auto_tag_slurm_array(self, tmp_db):
        """SLURM_ARRAY_JOB_ID should auto-add a sweep:<id> tag."""
        with patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "99999"}):
            logger = RegistryLogger(db_path=tmp_db, tags=["resnet"])
        assert "sweep:99999" in logger._tags
        assert "resnet" in logger._tags

    def test_auto_tag_no_duplicate(self, tmp_db):
        """If user already has the sweep tag, don't duplicate."""
        with patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "99999"}):
            logger = RegistryLogger(db_path=tmp_db, tags=["sweep:99999", "resnet"])
        assert logger._tags.count("sweep:99999") == 1

    def test_tags_and_notes(self, tmp_db):
        logger = RegistryLogger(
            db_path=tmp_db,
            tags=["ssl", "debug"],
            notes="Quick test run",
        )
        logger._run_id = "tagged-run"
        logger._run_dir = "/tmp/runs/tagged"
        logger.log_hyperparams({"lr": 0.01})

        run = logger._db.get_run("tagged-run")
        assert run["tags"] == ["ssl", "debug"]
        assert run["notes"] == "Quick test run"


# ============================================================================
# Query API
# ============================================================================


class TestRegistry:
    """Tests for the Registry query API."""

    def test_query_all(self, populated_db):
        reg = Registry(populated_db)
        runs = reg.query()
        assert len(runs) == 3
        assert all(isinstance(r, RunRecord) for r in runs)

    def test_query_by_tag(self, populated_db):
        reg = Registry(populated_db)
        runs = reg.query(tag="resnet")
        assert len(runs) == 2

    def test_query_by_sweep_tag(self, populated_db):
        reg = Registry(populated_db)
        runs = reg.query(tag="sweep:100")
        assert len(runs) == 2

    def test_query_by_hparams(self, populated_db):
        reg = Registry(populated_db)
        runs = reg.query(hparams={"lr": 0.01})
        assert len(runs) == 1
        assert runs[0].run_id == "run-001"

    def test_query_sort_by_summary(self, populated_db):
        reg = Registry(populated_db)
        runs = reg.query(tag="sweep:100", sort_by="summary.val_acc", descending=True)
        assert runs[0].run_id == "run-002"
        assert runs[0].summary["val_acc"] == 0.92

    def test_get(self, populated_db):
        reg = Registry(populated_db)
        run = reg.get("run-001")
        assert run is not None
        assert run.run_id == "run-001"
        assert run.summary["val_acc"] == 0.85
        assert run.tags == ["resnet", "baseline", "sweep:100"]
        assert run.notes == "First baseline run"

    def test_get_nonexistent(self, populated_db):
        reg = Registry(populated_db)
        assert reg.get("nonexistent") is None

    def test_getitem(self, populated_db):
        reg = Registry(populated_db)
        run = reg["run-001"]
        assert run.run_id == "run-001"

    def test_getitem_raises(self, populated_db):
        reg = Registry(populated_db)
        with pytest.raises(KeyError):
            reg["nonexistent"]

    def test_len(self, populated_db):
        reg = Registry(populated_db)
        assert len(reg) == 3

    def test_repr(self, populated_db):
        reg = Registry(populated_db)
        r = repr(reg)
        assert "Registry" in r
        assert "runs=3" in r

    def test_to_dataframe(self, populated_db):
        reg = Registry(populated_db)
        df = reg.to_dataframe(tag="sweep:100")
        assert len(df) == 2
        assert "summary.val_acc" in df.columns
        assert "hparams.lr" in df.columns
        assert "tags" in df.columns
        assert "notes" in df.columns

    def test_to_dataframe_empty(self, populated_db):
        reg = Registry(populated_db)
        df = reg.to_dataframe(tag="nonexistent")
        assert len(df) == 0


# ============================================================================
# open_registry
# ============================================================================


class TestOpenRegistry:
    """Tests for the open_registry() factory."""

    def test_open_with_explicit_path(self, tmp_db):
        db = _registry_db.RegistryDB(tmp_db)
        db.insert_run("test-run")
        db.close()

        reg = open_registry(tmp_db)
        assert len(reg) == 1
        reg.close()

    def test_open_without_path_raises(self):
        from stable_pretraining._config import get_config

        cfg = get_config()
        original = cfg.cache_dir
        cfg._cache_dir = None
        try:
            with pytest.raises(ValueError, match="No db_path provided"):
                open_registry()
        finally:
            cfg._cache_dir = original


# ============================================================================
# _flatten_params
# ============================================================================


class TestFlattenParams:
    """Tests for the _flatten_params helper."""

    def test_flat_dict(self):
        result = _flatten_params({"lr": 0.01, "epochs": 100})
        assert result == {"lr": 0.01, "epochs": 100}

    def test_nested_dict(self):
        result = _flatten_params({"optimizer": {"lr": 0.01, "wd": 1e-4}})
        assert result == {"optimizer.lr": 0.01, "optimizer.wd": 1e-4}

    def test_list_values(self):
        result = _flatten_params({"layers": [64, 128, 256]})
        assert result == {"layers.0": 64, "layers.1": 128, "layers.2": 256}

    def test_non_serializable_values(self):
        result = _flatten_params({"fn": lambda x: x})
        assert "fn" in result
        assert isinstance(result["fn"], str)


# ============================================================================
# CLI (spt registry ...)
# ============================================================================


class TestRegistryCLI:
    """Tests for the ``spt registry`` CLI commands."""

    @pytest.fixture
    def cli_db(self, tmp_db):
        """Populate a DB with sample runs and return the path."""
        db = _registry_db.RegistryDB(tmp_db)
        db.insert_run(
            "cli-run-1",
            status="completed",
            run_dir="/tmp/r1",
            config={"lr": 0.01},
            hparams={"lr": 0.01},
            tags=["resnet", "sweep:100"],
        )
        db.update_run("cli-run-1", summary={"val_acc": 0.85, "train_loss": 0.12})
        db.insert_run(
            "cli-run-2",
            status="completed",
            run_dir="/tmp/r2",
            config={"lr": 0.1},
            hparams={"lr": 0.1},
            tags=["resnet", "sweep:100"],
        )
        db.update_run("cli-run-2", summary={"val_acc": 0.92, "train_loss": 0.08})
        db.insert_run(
            "cli-run-3",
            status="running",
            run_dir="/tmp/r3",
            tags=["vit"],
        )
        db.close()
        return tmp_db

    def test_ls(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(app, ["registry", "ls", "--db", cli_db])
        assert result.exit_code == 0
        assert "cli-run-1" in result.output
        assert "cli-run-2" in result.output
        assert "cli-run-3" in result.output

    def test_ls_filter_by_tag(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "ls", "--db", cli_db, "--tag", "vit"]
        )
        assert result.exit_code == 0
        assert "cli-run-3" in result.output
        assert "cli-run-1" not in result.output

    def test_ls_filter_by_status(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "ls", "--db", cli_db, "--status", "completed"]
        )
        assert result.exit_code == 0
        assert "cli-run-1" in result.output
        assert "cli-run-3" not in result.output

    def test_show(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "show", "cli-run-1", "--db", cli_db]
        )
        assert result.exit_code == 0
        assert "cli-run-1" in result.output
        assert "val_acc" in result.output
        assert "0.85" in result.output
        assert "resnet" in result.output

    def test_show_not_found(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "show", "nonexistent", "--db", cli_db]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_best(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "best", "val_acc", "--db", cli_db]
        )
        assert result.exit_code == 0
        # cli-run-2 has higher val_acc (0.92), should be first
        lines = result.output.strip().split("\n")
        assert "cli-run-2" in lines[1]  # first data row

    def test_best_ascending(self, cli_db):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        result = CliRunner().invoke(
            app, ["registry", "best", "train_loss", "--asc", "--db", cli_db]
        )
        assert result.exit_code == 0
        # cli-run-2 has lower loss (0.08), should be first with --asc
        lines = result.output.strip().split("\n")
        assert "cli-run-2" in lines[1]

    def test_export_csv(self, cli_db, tmp_path):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        output = str(tmp_path / "export.csv")
        result = CliRunner().invoke(app, ["registry", "export", output, "--db", cli_db])
        assert result.exit_code == 0
        assert "Exported 3 runs" in result.output

        import pandas as pd

        df = pd.read_csv(output)
        assert len(df) == 3
        assert "summary.val_acc" in df.columns


# ============================================================================
# Hydra config flattening & injection (Manager integration)
# ============================================================================


class TestFlattenHydraConfig:
    """Tests for Manager._flatten_hydra_config (the shared flattening logic)."""

    def _make_manager_with_configs(self, trainer_cfg, module_cfg, data_cfg):
        """Create a Manager with DictConfig fields for testing flatten logic."""
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        # Manager stores raw configs when passed as dicts
        manager = Manager(
            trainer=OmegaConf.create(trainer_cfg),
            module=BoringModule(),  # already instantiated → skipped by flatten
            data=BoringDataModule(),  # already instantiated → skipped by flatten
        )
        # Override with DictConfigs so flatten picks them up
        if module_cfg is not None:
            manager.module = OmegaConf.create(module_cfg)
        if data_cfg is not None:
            manager.data = OmegaConf.create(data_cfg)
        return manager

    def test_basic_flat_keys(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 100, "accelerator": "gpu"},
            {"_target_": "my.Module", "lr": 0.01},
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["trainer.max_epochs"] == 100
        assert flat["trainer.accelerator"] == "gpu"
        assert flat["module._target_"] == "my.Module"
        assert flat["module.lr"] == 0.01

    def test_deeply_nested(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 10},
            {
                "optim": {
                    "optimizer": {
                        "type": "LARS",
                        "lr": 5.0,
                        "weight_decay": 1e-6,
                    },
                    "scheduler": {
                        "type": "CosineAnnealing",
                        "T_max": 100,
                        "eta_min": 1e-5,
                    },
                },
                "backbone": {
                    "_target_": "torchvision.models.resnet50",
                    "pretrained": False,
                },
            },
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["module.optim.optimizer.type"] == "LARS"
        assert flat["module.optim.optimizer.lr"] == 5.0
        assert flat["module.optim.optimizer.weight_decay"] == 1e-6
        assert flat["module.optim.scheduler.type"] == "CosineAnnealing"
        assert flat["module.optim.scheduler.T_max"] == 100
        assert flat["module.backbone._target_"] == "torchvision.models.resnet50"

    def test_lists_expanded(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 10},
            {"hidden_dims": [128, 256, 512]},
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["module.hidden_dims.0"] == 128
        assert flat["module.hidden_dims.1"] == 256
        assert flat["module.hidden_dims.2"] == 512
        assert "module.hidden_dims" not in flat  # list itself is removed

    def test_nested_lists(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 10},
            {
                "callbacks": [
                    {"_target_": "A", "k": 1},
                    {"_target_": "B", "k": 2},
                ]
            },
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["module.callbacks.0._target_"] == "A"
        assert flat["module.callbacks.0.k"] == 1
        assert flat["module.callbacks.1._target_"] == "B"

    def test_all_instantiated_returns_empty(self):
        """When trainer/module/data are all instantiated objects, returns {}."""
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import (
            BoringTrainer,
            BoringModule,
            BoringDataModule,
        )

        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        assert manager._flatten_hydra_config() == {}

    def test_mixed_instantiated_and_config(self):
        """Only DictConfig fields are flattened; instantiated objects are skipped."""
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        manager = Manager(
            trainer=OmegaConf.create({"max_epochs": 50, "accelerator": "cpu"}),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        flat = manager._flatten_hydra_config()
        assert "trainer.max_epochs" in flat
        assert not any(k.startswith("module.") for k in flat)
        assert not any(k.startswith("data.") for k in flat)


class TestInjectHydraHparams:
    """Tests that _inject_hydra_hparams puts flattened config into module.hparams."""

    def test_hparams_injected_into_module(self):
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        manager = Manager(
            trainer=OmegaConf.create(
                {
                    "max_epochs": 100,
                    "accelerator": "cpu",
                }
            ),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        # Simulate the instantiation path
        module = manager.instantiated_module

        manager._inject_hydra_hparams()

        assert "trainer.max_epochs" in module.hparams
        assert module.hparams["trainer.max_epochs"] == 100
        assert module.hparams["trainer.accelerator"] == "cpu"

    def test_deeply_nested_hparams_in_module(self):
        """Deeply nested Hydra config keys are flattened and injected correctly."""
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        # Build a Manager with deeply nested module config as DictConfig
        trainer_cfg = OmegaConf.create({"max_epochs": 10})
        module_cfg = OmegaConf.create(
            {
                "optim": {"optimizer": {"lr": 0.01, "wd": 1e-4}},
                "backbone": {"name": "resnet50", "layers": [3, 4, 6, 3]},
            }
        )
        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        # Override .module with DictConfig so flatten picks it up,
        # but keep the instantiated module available
        manager.module = module_cfg

        flat = manager._flatten_hydra_config()
        assert flat["module.optim.optimizer.lr"] == 0.01
        assert flat["module.optim.optimizer.wd"] == 1e-4
        assert flat["module.backbone.name"] == "resnet50"
        assert flat["module.backbone.layers.0"] == 3
        assert flat["module.backbone.layers.3"] == 3
        assert flat["trainer.max_epochs"] == 10


class TestEndToEndConfigToRegistry:
    """End-to-end: Manager → cache_dir → RegistryLogger + CSVLogger → queryable DB.

    These tests go through the Manager (the real production path) to verify
    that the full Hydra config is flattened, injected into module.hparams,
    and ends up queryable in the SQLite registry. All outputs land in
    tmp_path via cache_dir — nothing leaks to the working directory.
    """

    def _run_via_manager(self, tmp_path, trainer_cfg, module, data):
        """Run a training job through the Manager.

        cache_dir is already set to tmp_path by the autouse fixture.
        """
        from stable_pretraining.manager import Manager

        manager = Manager(trainer=trainer_cfg, module=module, data=data)
        manager()
        return str(tmp_path / "registry.db")

    def test_config_flows_through_manager_to_registry(self, tmp_path):
        """Full integration: Hydra config flows through Manager to the registry DB."""
        from omegaconf import OmegaConf
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        db_path = self._run_via_manager(
            tmp_path, trainer_cfg, BoringModule(), BoringDataModule()
        )

        reg = Registry(_registry_db.RegistryDB(db_path))
        assert len(reg) == 1
        run = reg.query()[0]
        assert run.status == "completed"
        assert run.hparams["trainer.max_epochs"] == 1
        assert run.hparams["trainer.accelerator"] == "cpu"
        assert run.run_dir is not None

    def test_deeply_nested_config_queryable(self, tmp_path):
        """Deeply nested Hydra keys survive the Manager → DB pipeline."""
        from omegaconf import OmegaConf
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        module = BoringModule()
        # Override module config with deeply nested structure
        # (Manager stores this as self.module DictConfig)

        db_path = self._run_via_manager(
            tmp_path, trainer_cfg, module, BoringDataModule()
        )

        reg = Registry(_registry_db.RegistryDB(db_path))
        run = reg.query()[0]
        # Trainer config keys are flattened
        assert "trainer.max_epochs" in run.hparams
        assert "trainer.accelerator" in run.hparams

    def test_no_files_leak_to_cwd(self, tmp_path):
        """With cache_dir set, nothing should be written to CWD."""
        import glob
        from omegaconf import OmegaConf
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        # Snapshot CWD before
        cwd_before = set(glob.glob("*"))

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        self._run_via_manager(tmp_path, trainer_cfg, BoringModule(), BoringDataModule())

        # Snapshot CWD after
        cwd_after = set(glob.glob("*"))
        leaked = cwd_after - cwd_before
        assert not leaked, f"Files leaked to CWD: {leaked}"

    def test_dataframe_has_flattened_hparams(self, tmp_path):
        """to_dataframe() exposes flattened hparams as columns."""
        from omegaconf import OmegaConf
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        db_path = self._run_via_manager(
            tmp_path, trainer_cfg, BoringModule(), BoringDataModule()
        )

        reg = Registry(_registry_db.RegistryDB(db_path))
        df = reg.to_dataframe()
        assert len(df) == 1
        assert "hparams.trainer.max_epochs" in df.columns
        assert df.iloc[0]["hparams.trainer.max_epochs"] == 1
