"""Manager FSDP detection + logging.

Verifies:
- :func:`is_fsdp_strategy` correctly identifies FSDP strategies (and trainers
  wrapping them) without raising on plain objects.
- :func:`describe_fsdp_strategy` returns the expected summary dict.
- :meth:`Manager._log_fsdp_info` is a no-op for non-FSDP strategies and emits
  the expected log lines (and warning for ``TeacherStudentWrapper``) for FSDP.

These tests are CPU-runnable: no FSDP wrap or distributed initialization is
required to exercise the detection / logging logic.
"""

from __future__ import annotations

import logging as stdlib_logging

import pytest
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.utils.fsdp import (
    describe_fsdp_strategy,
    is_fsdp_strategy,
    make_fsdp_strategy,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# is_fsdp_strategy / describe_fsdp_strategy
# ---------------------------------------------------------------------------


def test_is_fsdp_strategy_handles_plain_objects():
    assert is_fsdp_strategy(None) is False
    assert is_fsdp_strategy("ddp") is False
    assert is_fsdp_strategy(object()) is False


def test_is_fsdp_strategy_recognizes_our_strategy():
    strat = make_fsdp_strategy()
    assert is_fsdp_strategy(strat) is True


def test_is_fsdp_strategy_recognizes_via_trainer_attribute():
    """If passed an object with a ``.strategy`` attribute (i.e. a Trainer),
    inspect that attribute."""

    class _FakeTrainer:
        strategy = make_fsdp_strategy()

    assert is_fsdp_strategy(_FakeTrainer()) is True


def test_describe_fsdp_strategy_for_non_fsdp():
    assert describe_fsdp_strategy(None) == {"is_fsdp": False}
    assert describe_fsdp_strategy("ddp") == {"is_fsdp": False}


def test_describe_fsdp_strategy_for_our_strategy():
    strat = make_fsdp_strategy(
        sharding_strategy="FULL_SHARD",
        state_dict_type="sharded",
    )
    info = describe_fsdp_strategy(strat)
    assert info["is_fsdp"] is True
    assert info["subclass"] == "CallbackAwareFSDPStrategy"
    assert "FULL_SHARD" in info["sharding_strategy"]
    assert "sharded" in info["state_dict_type"]
    assert info["n_ignored_modules"] == 0


# ---------------------------------------------------------------------------
# Manager._log_fsdp_info — verify it is a no-op for non-FSDP and fires for FSDP
# ---------------------------------------------------------------------------


def _supervised_forward(self, batch, stage):
    out = self.backbone(batch["image"])
    return {"embedding": out, "label": batch["label"], "loss": out.sum() * 0.0}


def _build_dummy_manager(strategy):
    """Build a Manager with a trainer mock that exposes ``.strategy``.

    We don't run ``Manager.__call__``; we just want to invoke
    ``_log_fsdp_info`` directly.
    """

    class _FakeTrainer:
        pass

    trainer = _FakeTrainer()
    trainer.strategy = strategy

    mgr = spt.Manager.__new__(spt.Manager)
    mgr._trainer = trainer
    return mgr


def test_log_fsdp_info_noop_for_non_fsdp(caplog):
    """No FSDP-specific log lines when the strategy is not FSDP."""
    mgr = _build_dummy_manager(strategy="ddp")
    with caplog.at_level(stdlib_logging.INFO):
        mgr._log_fsdp_info(needs_teacher_student=False)
    assert "FSDP STRATEGY DETECTED" not in caplog.text


def test_log_fsdp_info_logs_for_fsdp(caplog):
    """FSDP strategy triggers info logs with key configuration values."""
    strat = make_fsdp_strategy()
    mgr = _build_dummy_manager(strategy=strat)
    # loguru -> stdlib bridge: capture from the loguru sink Manager uses.
    # caplog won't see loguru by default; instead snoop on logger.add().
    from loguru import logger

    captured = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    try:
        mgr._log_fsdp_info(needs_teacher_student=False)
    finally:
        logger.remove(sink_id)

    text = "\n".join(captured)
    assert "FSDP STRATEGY DETECTED" in text
    assert "subclass: CallbackAwareFSDPStrategy" in text
    assert "sharding_strategy" in text
    assert "state_dict_type" in text


def test_log_fsdp_info_warns_when_teacher_student_present():
    """The TS-alignment warning fires only when needs_teacher_student=True."""
    strat = make_fsdp_strategy()
    mgr = _build_dummy_manager(strategy=strat)
    from loguru import logger

    captured_warn = []
    sink_id = logger.add(
        lambda msg: captured_warn.append(str(msg)), level="WARNING"
    )
    try:
        # No TS -> no warning
        mgr._log_fsdp_info(needs_teacher_student=False)
        assert all(
            "TeacherStudentWrapper" not in line for line in captured_warn
        ), captured_warn
        # Reset
        captured_warn.clear()
        # With TS -> one warning line
        mgr._log_fsdp_info(needs_teacher_student=True)
        warn_text = "\n".join(captured_warn)
        assert "TeacherStudentWrapper" in warn_text
        assert "identical policies" in warn_text
    finally:
        logger.remove(sink_id)


def test_log_fsdp_info_describes_ignored_modules_count():
    """The ignored_modules count tracks what the strategy holds."""

    class _FakeTrainer:
        pass

    trainer = _FakeTrainer()
    strat = make_fsdp_strategy()
    # Manually inject ignored_modules to mimic post-_setup_model state.
    sentinel_a = nn.Linear(2, 2)
    sentinel_b = nn.Linear(2, 2)
    strat.kwargs["ignored_modules"] = [sentinel_a, sentinel_b]
    trainer.strategy = strat

    info = describe_fsdp_strategy(trainer)
    assert info["n_ignored_modules"] == 2
