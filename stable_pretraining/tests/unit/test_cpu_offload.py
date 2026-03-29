"""Unit tests for CPUOffloadCallback."""

import pytest
import torch
from unittest.mock import Mock, patch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from stable_pretraining.callbacks import CPUOffloadCallback

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_trainer_ddp():
    trainer = Mock(spec=Trainer)
    trainer.strategy = DDPStrategy()
    trainer.global_rank = 0
    trainer.world_size = 4
    trainer.global_step = 1000
    trainer.current_epoch = 2
    return trainer


@pytest.fixture
def mock_trainer_single_device():
    trainer = Mock(spec=Trainer)
    trainer.strategy = SingleDeviceStrategy(device=torch.device("cpu"))
    trainer.global_rank = 0
    trainer.world_size = 1
    trainer.global_step = 500
    trainer.current_epoch = 1
    return trainer


@pytest.fixture
def mock_trainer_fsdp():
    trainer = Mock(spec=Trainer)
    fsdp_strategy = Mock()
    fsdp_strategy.__class__.__name__ = "FSDPStrategy"
    trainer.strategy = fsdp_strategy
    trainer.global_rank = 0
    trainer.world_size = 4
    trainer.global_step = 1000
    trainer.current_epoch = 2
    return trainer


@pytest.fixture
def mock_pl_module():
    module = Mock()
    param1 = torch.randn(100, 100)
    param2 = torch.randn(200, 200)
    param1.requires_grad = True
    param2.requires_grad = True
    module.parameters.return_value = [param1, param2]
    return module


@pytest.fixture
def sample_checkpoint():
    return {
        "state_dict": {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(20, 10),
        },
        "optimizer_states": [
            {
                "state": {
                    0: {
                        "momentum": torch.randn(10, 10),
                        "exp_avg": torch.randn(10, 10),
                    },
                    1: {"momentum": torch.randn(10), "exp_avg": torch.randn(10)},
                },
                "param_groups": [{"lr": 0.001, "weight_decay": 0.01}],
            }
        ],
        "lr_schedulers": [{"last_epoch": 100, "base_lr": 0.001}],
        "epoch": 2,
        "global_step": 1000,
    }


# ============================================================================
# Strategy Compatibility
# ============================================================================


def test_strategy_compatibility_ddp(mock_trainer_ddp, mock_pl_module):
    callback = CPUOffloadCallback()
    callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
    assert callback._is_enabled is True


def test_strategy_compatibility_single_device(
    mock_trainer_single_device, mock_pl_module
):
    callback = CPUOffloadCallback()
    callback.setup(mock_trainer_single_device, mock_pl_module, "fit")
    assert callback._is_enabled is True


def test_strategy_compatibility_fsdp(mock_trainer_fsdp, mock_pl_module):
    callback = CPUOffloadCallback()
    callback.setup(mock_trainer_fsdp, mock_pl_module, "fit")
    assert callback._is_enabled is False


def test_strategy_compatibility_unknown():
    trainer = Mock(spec=Trainer)
    unknown_strategy = Mock()
    unknown_strategy.__class__.__name__ = "UnknownStrategy"
    trainer.strategy = unknown_strategy
    trainer.global_rank = 0

    callback = CPUOffloadCallback()
    callback.setup(trainer, Mock(), "fit")
    assert callback._is_enabled is False


# ============================================================================
# Initialization
# ============================================================================


def test_init_default_params():
    callback = CPUOffloadCallback()
    assert callback.offload_keys == ["state_dict", "optimizer_states", "lr_schedulers"]
    assert callback._checkpoint_count == 0
    assert callback._total_time == 0.0
    assert callback._total_memory_freed == 0.0


def test_init_custom_params():
    callback = CPUOffloadCallback(offload_keys=["state_dict"])
    assert callback.offload_keys == ["state_dict"]


# ============================================================================
# _to_cpu_recursive
# ============================================================================


def test_to_cpu_recursive_tensor():
    callback = CPUOffloadCallback()
    parent = {"t": torch.randn(10, 10)}
    moved, nbytes = callback._to_cpu_recursive(parent["t"], parent, "t")
    # CPU tensors don't count as moved (not is_cuda)
    assert moved == 0
    assert parent["t"].device == torch.device("cpu")


def test_to_cpu_recursive_dict():
    callback = CPUOffloadCallback()
    data = {
        "tensor1": torch.randn(5, 5),
        "nested": {"tensor2": torch.randn(2, 2)},
    }
    parent = {"key": data}
    moved, nbytes = callback._to_cpu_recursive(data, parent, "key")
    assert data["tensor1"].device == torch.device("cpu")
    assert data["nested"]["tensor2"].device == torch.device("cpu")


def test_to_cpu_recursive_list():
    callback = CPUOffloadCallback()
    data = [torch.randn(2, 2), torch.randn(3, 3)]
    parent = {"key": data}
    moved, nbytes = callback._to_cpu_recursive(data, parent, "key")
    assert data[0].device == torch.device("cpu")
    assert data[1].device == torch.device("cpu")


def test_to_cpu_recursive_tuple():
    callback = CPUOffloadCallback()
    data = (torch.randn(2, 2), torch.randn(3, 3))
    parent = {"key": data}
    moved, nbytes = callback._to_cpu_recursive(data, parent, "key")
    # Tuple gets converted to list internally, parent updated
    assert parent["key"][0].device == torch.device("cpu")
    assert parent["key"][1].device == torch.device("cpu")


def test_to_cpu_recursive_primitives():
    callback = CPUOffloadCallback()
    parent = {"a": 42, "b": 3.14, "c": "hello", "d": True, "e": None}
    for k in parent:
        moved, nbytes = callback._to_cpu_recursive(parent[k], parent, k)
        assert moved == 0
        assert nbytes == 0


def test_to_cpu_recursive_custom_object():
    callback = CPUOffloadCallback()
    custom = Mock()
    parent = {"obj": custom}
    moved, nbytes = callback._to_cpu_recursive(custom, parent, "obj")
    assert moved == 0
    assert parent["obj"] is custom  # unchanged


def test_to_cpu_recursive_mixed_list():
    callback = CPUOffloadCallback()
    data = [torch.randn(2, 2), 42, "string", Mock(), torch.randn(3, 3)]
    parent = {"key": data}
    moved, nbytes = callback._to_cpu_recursive(data, parent, "key")
    assert data[0].device == torch.device("cpu")
    assert data[1] == 42
    assert data[2] == "string"
    assert data[4].device == torch.device("cpu")


def test_to_cpu_recursive_deeply_nested():
    callback = CPUOffloadCallback()
    data = {"l1": {"l2": {"l3": {"tensor": torch.randn(5, 5)}}}}
    parent = {"key": data}
    callback._to_cpu_recursive(data, parent, "key")
    assert data["l1"]["l2"]["l3"]["tensor"].device == torch.device("cpu")


# ============================================================================
# on_save_checkpoint
# ============================================================================


def test_on_save_checkpoint_enabled(
    mock_trainer_ddp, mock_pl_module, sample_checkpoint
):
    callback = CPUOffloadCallback()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=8e9),
    ):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, sample_checkpoint)

    assert callback._checkpoint_count == 1
    assert callback._total_time > 0
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == torch.device(
        "cpu"
    )


def test_on_save_checkpoint_disabled(
    mock_trainer_fsdp, mock_pl_module, sample_checkpoint
):
    callback = CPUOffloadCallback()
    callback.setup(mock_trainer_fsdp, mock_pl_module, "fit")

    original_device = sample_checkpoint["state_dict"]["layer1.weight"].device
    callback.on_save_checkpoint(mock_trainer_fsdp, mock_pl_module, sample_checkpoint)

    assert callback._checkpoint_count == 0
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == original_device


def test_on_save_checkpoint_non_rank_zero(
    mock_trainer_ddp, mock_pl_module, sample_checkpoint
):
    mock_trainer_ddp.global_rank = 1
    callback = CPUOffloadCallback()
    callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
    callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, sample_checkpoint)
    assert callback._checkpoint_count == 0


def test_on_save_checkpoint_empty():
    callback = CPUOffloadCallback()
    callback._is_enabled = True
    trainer = Mock(spec=Trainer)
    trainer.global_rank = 0

    with patch("torch.cuda.is_available", return_value=False):
        callback.on_save_checkpoint(trainer, Mock(), {})

    assert callback._checkpoint_count == 1


# ============================================================================
# State Dict
# ============================================================================


def test_state_dict_save():
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 5
    callback._total_time = 25.5
    callback._total_memory_freed = 40.2
    callback._is_enabled = True

    state = callback.state_dict()

    assert state["checkpoint_count"] == 5
    assert state["total_time"] == 25.5
    assert state["total_memory_freed"] == 40.2
    assert state["is_enabled"] is True


def test_state_dict_load():
    callback = CPUOffloadCallback()

    state = {
        "checkpoint_count": 10,
        "total_time": 50.0,
        "total_memory_freed": 80.0,
        "is_enabled": False,
    }

    callback.load_state_dict(state)

    assert callback._checkpoint_count == 10
    assert callback._total_time == 50.0
    assert callback._total_memory_freed == 80.0
    assert callback._is_enabled is False


# ============================================================================
# Lifecycle Hooks
# ============================================================================


def test_on_exception(mock_trainer_ddp, mock_pl_module):
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 3
    callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
    # Should not raise
    callback.on_exception(mock_trainer_ddp, mock_pl_module, RuntimeError("Test"))


def test_cumulative_statistics(mock_trainer_ddp, mock_pl_module):
    callback = CPUOffloadCallback()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=1e9),
    ):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")

        ckpt1 = {"state_dict": {"w": torch.randn(10, 10)}}
        ckpt2 = {"state_dict": {"w": torch.randn(10, 10)}}

        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, ckpt1)
        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, ckpt2)

    assert callback._checkpoint_count == 2
    assert callback._total_time > 0
