"""Unit tests for flow matching solver utilities.

Run with: pytest test_solver.py -v -m unit
"""

import math
from typing import Callable

import pytest
import torch
from torch import Tensor

from stable_pretraining.utils.solver import (
    ODESolver,
    flow_matching_sample,
    _build_time_schedule,
    _step_euler,
    _step_midpoint,
    _step_heun,
    _step_rk4,
    _step_dpm_2,
    _step_dpm_3,
)


# === Fixtures ===


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


@pytest.fixture
def generator() -> torch.Generator:
    """Fixed generator for reproducibility."""
    return torch.Generator().manual_seed(42)


@pytest.fixture
def simple_velocity_fn() -> Callable[[Tensor, Tensor], Tensor]:
    """Trivial velocity field: v(x, t) = -x.

    ODE: dx/dt = -x has solution x(t) = x(0) * exp(-t).
    Integrating from t=0 to t=1: x(1) = x(0) * exp(-1).
    """

    def v_fn(x: Tensor, t: Tensor) -> Tensor:
        return -x

    return v_fn


@pytest.fixture
def constant_velocity_fn() -> Callable[[Tensor, Tensor], Tensor]:
    """Constant velocity field: v(x, t) = 1.

    ODE: dx/dt = 1 has solution x(t) = x(0) + t.
    Integrating from t=0 to t=1: x(1) = x(0) + 1.
    """

    def v_fn(x: Tensor, t: Tensor) -> Tensor:
        return torch.ones_like(x)

    return v_fn


@pytest.fixture
def linear_velocity_fn() -> Callable[[Tensor, Tensor], Tensor]:
    """Linear interpolation velocity (typical flow matching): v(x, t) = (target - x) / (1 - t + eps).

    Simplified version: v(x, t) = target - noise (constant velocity toward target).
    """
    target = torch.ones(1)  # Target is all ones

    def v_fn(x: Tensor, t: Tensor) -> Tensor:
        # Constant velocity pointing toward target
        return target.expand_as(x) - x / (t.view(-1, *([1] * (x.ndim - 1))) + 1.0)

    return v_fn


# === Time Schedule Tests ===


@pytest.mark.unit
class TestBuildTimeSchedule:
    """Base test class."""

    def test_linear_schedule_endpoints(self, device, dtype):
        """Linear schedule should span [0, 1]."""
        t = _build_time_schedule(10, "linear", device, dtype)

        assert t.shape == (11,)
        assert t[0].item() == pytest.approx(0.0)
        assert t[-1].item() == pytest.approx(1.0)

    def test_linear_schedule_uniform_spacing(self, device, dtype):
        """Linear schedule should have uniform spacing."""
        t = _build_time_schedule(10, "linear", device, dtype)
        diffs = t[1:] - t[:-1]

        assert torch.allclose(diffs, diffs[0].expand_as(diffs))

    def test_cosine_schedule_endpoints(self, device, dtype):
        """Cosine schedule should span [0, 1]."""
        t = _build_time_schedule(10, "cosine", device, dtype)

        assert t[0].item() == pytest.approx(0.0)
        assert t[-1].item() == pytest.approx(1.0)

    def test_cosine_schedule_more_steps_at_boundaries(self, device, dtype):
        """Cosine schedule should have smaller steps near 0 and 1."""
        t = _build_time_schedule(20, "cosine", device, dtype)
        diffs = t[1:] - t[:-1]

        # Steps at boundaries should be smaller than in the middle
        assert diffs[0] < diffs[10]
        assert diffs[-1] < diffs[10]

    def test_quadratic_schedule_endpoints(self, device, dtype):
        """Quadratic schedule should span [0, 1]."""
        t = _build_time_schedule(10, "quadratic", device, dtype)

        assert t[0].item() == pytest.approx(0.0)
        assert t[-1].item() == pytest.approx(1.0)

    def test_quadratic_schedule_more_steps_near_end(self, device, dtype):
        """Quadratic schedule should have more steps near t=1."""
        t = _build_time_schedule(20, "quadratic", device, dtype)
        diffs = t[1:] - t[:-1]

        # Steps near t=1 should be larger (values more spread in linear space)
        assert diffs[-1] > diffs[0]

    def test_invalid_schedule_raises(self, device, dtype):
        """Unknown schedule should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            _build_time_schedule(10, "invalid_schedule", device, dtype)


# === Individual Solver Step Tests ===


@pytest.mark.unit
class TestSolverSteps:
    """Test individual solver step functions for correctness."""

    @pytest.fixture
    def setup(self, device, dtype):
        """Common setup for step tests."""
        x = torch.randn(4, 8, 16, device=device, dtype=dtype)
        t = torch.full((4,), 0.5, device=device, dtype=dtype)
        dt = torch.tensor(0.1, device=device, dtype=dtype)
        t_next = t + dt
        return x, t, dt, t_next

    def test_euler_step_shape(self, setup, constant_velocity_fn):
        """Euler step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_euler(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_euler_constant_velocity(self, setup, constant_velocity_fn):
        """Euler with constant velocity: x_new = x + dt * 1."""
        x, t, dt, t_next = setup
        x_new = _step_euler(constant_velocity_fn, x, t, dt, t_next)

        expected = x + dt
        assert torch.allclose(x_new, expected)

    def test_midpoint_step_shape(self, setup, constant_velocity_fn):
        """Midpoint step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_midpoint(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_heun_step_shape(self, setup, constant_velocity_fn):
        """Heun step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_heun(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_heun_constant_velocity(self, setup, constant_velocity_fn):
        """Heun with constant velocity should equal Euler."""
        x, t, dt, t_next = setup
        x_euler = _step_euler(constant_velocity_fn, x, t, dt, t_next)
        x_heun = _step_heun(constant_velocity_fn, x, t, dt, t_next)

        assert torch.allclose(x_euler, x_heun)

    def test_rk4_step_shape(self, setup, constant_velocity_fn):
        """RK4 step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_rk4(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_dpm2_step_shape(self, setup, constant_velocity_fn):
        """DPM-2 step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_dpm_2(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_dpm3_step_shape(self, setup, constant_velocity_fn):
        """DPM-3 step should preserve shape."""
        x, t, dt, t_next = setup
        x_new = _step_dpm_3(constant_velocity_fn, x, t, dt, t_next)

        assert x_new.shape == x.shape

    def test_higher_order_more_accurate(self, simple_velocity_fn, device, dtype):
        """Higher-order solvers should be more accurate for smooth ODEs."""
        # For v(x,t) = -x, solution is x(t) = x(0) * exp(-t)
        x = torch.ones(1, 1, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=dtype)
        dt = torch.tensor(0.5, device=device, dtype=dtype)
        t_next = t + dt

        expected = x * math.exp(-0.5)

        x_euler = _step_euler(simple_velocity_fn, x, t, dt, t_next)
        x_heun = _step_heun(simple_velocity_fn, x, t, dt, t_next)
        x_rk4 = _step_rk4(simple_velocity_fn, x, t, dt, t_next)

        error_euler = (x_euler - expected).abs().item()
        error_heun = (x_heun - expected).abs().item()
        error_rk4 = (x_rk4 - expected).abs().item()

        assert error_rk4 < error_heun < error_euler


# === Flow Matching Sample Tests ===


@pytest.mark.unit
class TestFlowMatchingSample:
    """Base test class."""

    def test_output_shape(self, constant_velocity_fn, device, dtype, generator):
        """Output should match requested shape."""
        shape = (2, 16, 32)
        x = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=shape,
            num_steps=10,
            solver="euler",
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert x.shape == shape

    def test_output_dtype(self, constant_velocity_fn, device, generator):
        """Output should match requested dtype."""
        for test_dtype in [torch.float32, torch.float64]:
            x = flow_matching_sample(
                velocity_fn=constant_velocity_fn,
                shape=(2, 8, 8),
                num_steps=5,
                solver="euler",
                device=device,
                dtype=test_dtype,
                generator=generator,
            )
            assert x.dtype == test_dtype

    @pytest.mark.parametrize("solver", list(ODESolver))
    def test_all_solvers_run(
        self, constant_velocity_fn, solver, device, dtype, generator
    ):
        """All solvers should run without error."""
        x = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(2, 4, 4),
            num_steps=10,
            solver=solver,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert x.shape == (2, 4, 4)
        assert torch.isfinite(x).all()

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "quadratic"])
    def test_all_schedules_run(
        self, constant_velocity_fn, schedule, device, dtype, generator
    ):
        """All time schedules should run without error."""
        x = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(2, 4, 4),
            num_steps=10,
            solver="euler",
            time_schedule=schedule,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert x.shape == (2, 4, 4)
        assert torch.isfinite(x).all()

    def test_reproducibility_with_generator(self, constant_velocity_fn, device, dtype):
        """Same generator seed should produce identical results."""
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        x1 = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(4, 8, 8),
            num_steps=10,
            device=device,
            dtype=dtype,
            generator=gen1,
        )
        x2 = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(4, 8, 8),
            num_steps=10,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        assert torch.allclose(x1, x2)

    def test_different_seeds_different_results(
        self, constant_velocity_fn, device, dtype
    ):
        """Different seeds should produce different initial noise."""
        gen1 = torch.Generator().manual_seed(1)
        gen2 = torch.Generator().manual_seed(2)

        # Use zero velocity to just return the noise
        def zero_velocity(x, t):
            return torch.zeros_like(x)

        x1 = flow_matching_sample(
            velocity_fn=zero_velocity,
            shape=(4, 8, 8),
            num_steps=1,
            solver="euler",
            device=device,
            dtype=dtype,
            generator=gen1,
        )
        x2 = flow_matching_sample(
            velocity_fn=zero_velocity,
            shape=(4, 8, 8),
            num_steps=1,
            solver="euler",
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        assert not torch.allclose(x1, x2)

    def test_constant_velocity_integration(self, device, dtype, generator):
        """Constant velocity v=1: x(1) = x(0) + 1."""

        def v_fn(x, t):
            return torch.ones_like(x)

        # With many steps, should be accurate
        x = flow_matching_sample(
            velocity_fn=v_fn,
            shape=(1, 1, 1),
            num_steps=100,
            solver="rk4",
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # x(0) was noise, x(1) = x(0) + 1
        # We can't check exact value, but we can verify it changed by ~1
        gen_copy = torch.Generator().manual_seed(42)
        x0 = torch.randn(1, 1, 1, generator=gen_copy, device=device, dtype=dtype)

        assert (x - x0 - 1.0).abs().item() < 0.01

    def test_clamp_range(self, device, dtype, generator):
        """Clamping should keep values in range."""

        def large_velocity(x, t):
            return torch.ones_like(x) * 1000.0  # Would push x very high

        x = flow_matching_sample(
            velocity_fn=large_velocity,
            shape=(4, 8, 8),
            num_steps=10,
            solver="euler",
            clamp_range=(-1.0, 1.0),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        assert x.min() >= -1.0
        assert x.max() <= 1.0

    def test_return_trajectory(self, constant_velocity_fn, device, dtype, generator):
        """Should return trajectory when requested."""
        num_steps = 10
        result = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(2, 4, 4),
            num_steps=num_steps,
            solver="euler",
            device=device,
            dtype=dtype,
            generator=generator,
            return_trajectory=True,
        )

        assert isinstance(result, tuple)
        x, trajectory = result
        assert len(trajectory) == num_steps + 1  # Initial + each step
        assert all(t.shape == x.shape for t in trajectory)

    def test_cfg_requires_cond_fn(self, constant_velocity_fn, device, dtype, generator):
        """Using guidance_scale without cond_velocity_fn should raise."""
        with pytest.raises(ValueError, match="cond_velocity_fn required"):
            flow_matching_sample(
                velocity_fn=constant_velocity_fn,
                shape=(2, 4, 4),
                num_steps=5,
                guidance_scale=2.0,
                cond_velocity_fn=None,
                device=device,
                dtype=dtype,
                generator=generator,
            )

    def test_cfg_interpolation(self, device, dtype, generator):
        """CFG should interpolate between uncond and cond velocities."""

        def uncond_v(x, t):
            return torch.zeros_like(x)

        def cond_v(x, t):
            return torch.ones_like(x)

        # With scale=1, should act like cond_v
        x_cond = flow_matching_sample(
            velocity_fn=uncond_v,
            shape=(1, 1, 1),
            num_steps=10,
            solver="euler",
            guidance_scale=1.0,
            cond_velocity_fn=cond_v,
            device=device,
            dtype=dtype,
            generator=torch.Generator().manual_seed(42),
        )

        x_only_cond = flow_matching_sample(
            velocity_fn=cond_v,
            shape=(1, 1, 1),
            num_steps=10,
            solver="euler",
            device=device,
            dtype=dtype,
            generator=torch.Generator().manual_seed(42),
        )

        assert torch.allclose(x_cond, x_only_cond, atol=1e-5)

    def test_solver_string_enum_equivalence(self, constant_velocity_fn, device, dtype):
        """String and enum solver specs should produce identical results."""
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        x_str = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(2, 4, 4),
            num_steps=10,
            solver="heun",
            device=device,
            dtype=dtype,
            generator=gen1,
        )
        x_enum = flow_matching_sample(
            velocity_fn=constant_velocity_fn,
            shape=(2, 4, 4),
            num_steps=10,
            solver=ODESolver.HEUN,
            device=device,
            dtype=dtype,
            generator=gen2,
        )

        assert torch.allclose(x_str, x_enum)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
