"""Debug benchmark to identify bottlenecks."""

import torch
import timm
from stable_pretraining.backbone import EfficientMaskedTimmViT


def profile_forward_pass(model, x, num_iter=100):
    """Profile the forward pass."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()

    # Profile with events
    times = []

    with torch.no_grad():
        for _ in range(num_iter):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(x)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def test_component_breakdown(vit, x):
    """Break down time spent in each component."""
    vit.eval()

    times = {}

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = vit(x)
        torch.cuda.synchronize()

        # Patch embedding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            patches = vit.patch_embed(x)
        end.record()
        torch.cuda.synchronize()
        times["patch_embed"] = start.elapsed_time(end) / 100

        # Get a sample patches tensor for rest of ops
        patches = vit.patch_embed(x)
        B = patches.shape[0]

        # Add tokens
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            cls_tokens = vit.cls_token.expand(B, -1, -1)
            patches_with_cls = torch.cat([cls_tokens, patches], dim=1)
        else:
            patches_with_cls = patches

        # Pos embed
        start.record()
        for _ in range(100):
            x_pos = patches_with_cls + vit.pos_embed
            if hasattr(vit, "pos_drop") and vit.pos_drop is not None:
                x_pos = vit.pos_drop(x_pos)
        end.record()
        torch.cuda.synchronize()
        times["pos_embed"] = start.elapsed_time(end) / 100

        # Transformer blocks
        x_blocks = patches_with_cls + vit.pos_embed
        if hasattr(vit, "pos_drop") and vit.pos_drop is not None:
            x_blocks = vit.pos_drop(x_blocks)

        start.record()
        for _ in range(100):
            x_temp = x_blocks.clone()
            for blk in vit.blocks:
                x_temp = blk(x_temp)
        end.record()
        torch.cuda.synchronize()
        times["blocks"] = start.elapsed_time(end) / 100

        # Norm
        x_norm = x_blocks.clone()
        for blk in vit.blocks:
            x_norm = blk(x_norm)

        start.record()
        for _ in range(100):
            x_temp = vit.norm(x_norm)
        end.record()
        torch.cuda.synchronize()
        times["norm"] = start.elapsed_time(end) / 100

        # Head
        x_normed = vit.norm(x_norm)
        start.record()
        for _ in range(100):
            if hasattr(vit, "head"):
                _ = vit.head(x_normed[:, 0])
            else:
                _ = x_normed
        end.record()
        torch.cuda.synchronize()
        times["head"] = start.elapsed_time(end) / 100

    return times


def detailed_profile():
    """Run detailed profiling."""
    device = torch.device("cuda")
    model_name = "vit_tiny_patch16_224"

    print("=" * 80)
    print(f"Detailed GPU Profiling - {model_name}")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    vit = timm.create_model(model_name, pretrained=False).to(device)
    masked_vit = EfficientMaskedTimmViT(vit).to(device)

    B = 32
    x = torch.randn(B, 3, 224, 224, device=device)

    # Get patches
    with torch.no_grad():
        patches = vit.patch_embed(x)

    N = patches.shape[1]
    print(f"\nBatch size: {B}")
    print(f"Total patches: {N}")
    print()

    # Component breakdown for original model
    print("=" * 80)
    print("Component Breakdown (Original ViT)")
    print("=" * 80)
    component_times = test_component_breakdown(vit, x)
    total_component = sum(component_times.values())

    for component, time_ms in component_times.items():
        pct = (time_ms / total_component) * 100
        print(f"{component:20s}: {time_ms:6.2f} ms ({pct:5.1f}%)")
    print(f"{'Total':20s}: {total_component:6.2f} ms")

    # Test different masking ratios
    ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    results = {}

    print("\n" + "=" * 80)
    print("Masking Performance")
    print("=" * 80)

    for ratio in ratios:
        num_masked = int(N * ratio)
        num_kept = N - num_masked

        # Create masked patches - SAME PATTERN
        test_patches = patches.clone()
        if ratio > 0:
            mask_indices = torch.randperm(N)[:num_masked]
            test_patches[:, mask_indices, :] = float("nan")

        # Profile original model (only at 0%)
        if ratio == 0.0:
            time_orig = profile_forward_pass(vit, x)
            time_masked = profile_forward_pass(masked_vit, test_patches)
            overhead = ((time_masked / time_orig) - 1.0) * 100

            print("\n0% masking (baseline):")
            print(f"  Original model:     {time_orig:.2f} ms")
            print(f"  Masked model:       {time_masked:.2f} ms")
            print(f"  Wrapper overhead:   {overhead:+.1f}%")

            results[ratio] = {"same": time_masked, "diff": time_masked}
        else:
            time_same = profile_forward_pass(masked_vit, test_patches)

            # Create different pattern
            test_patches_diff = patches.clone()
            for i in range(B):
                mask_indices = torch.randperm(N)[:num_masked]
                test_patches_diff[i, mask_indices, :] = float("nan")

            time_diff = profile_forward_pass(masked_vit, test_patches_diff)

            results[ratio] = {"same": time_same, "diff": time_diff}

            speedup_same = results[0.0]["same"] / time_same
            speedup_diff = results[0.0]["diff"] / time_diff
            overhead_pct = ((time_diff / time_same) - 1.0) * 100

            print(f"\n{ratio * 100:.0f}% masking ({num_kept}/{N} patches):")
            print(
                f"  Same pattern:       {time_same:6.2f} ms (speedup: {speedup_same:.2f}x)"
            )
            print(
                f"  Different pattern:  {time_diff:6.2f} ms (speedup: {speedup_diff:.2f}x)"
            )
            print(f"  Pattern overhead:   {overhead_pct:+.1f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    baseline = results[0.0]["same"]
    print(
        f"\n{'Ratio':<10} {'Kept':<10} {'Same (ms)':<12} {'Speedup':<10} {'Theoretical':<12}"
    )
    print("-" * 80)

    for ratio in ratios:
        num_kept = N - int(N * ratio)
        time_ms = results[ratio]["same"]
        speedup = baseline / time_ms
        theoretical = 1.0 / (1.0 - ratio) if ratio < 1.0 else float("inf")
        theoretical_str = f"{theoretical:.2f}x" if theoretical < 100 else "inf"

        print(
            f"{ratio * 100:<10.0f} {num_kept:<10} {time_ms:<12.2f} {speedup:<10.2f} {theoretical_str:<12}"
        )

    # Efficiency analysis
    print("\n" + "=" * 80)
    print("Efficiency Analysis")
    print("=" * 80)
    print()

    for ratio in [0.25, 0.5, 0.75, 0.9]:
        if ratio in results:
            time_ms = results[ratio]["same"]
            actual_speedup = baseline / time_ms
            theoretical = 1.0 / (1.0 - ratio)
            efficiency = (actual_speedup / theoretical) * 100

            num_kept = N - int(N * ratio)
            print(
                f"{ratio * 100:>3.0f}% masking ({num_kept:>3} patches): "
                f"Efficiency: {efficiency:5.1f}% "
                f"(actual: {actual_speedup:.2f}x, theoretical: {theoretical:.2f}x)"
            )

    # PyTorch profiler
    print("\n" + "=" * 80)
    print("Detailed CUDA Profiler (90% masking)")
    print("=" * 80)

    test_patches_90 = patches.clone()
    mask_indices = torch.randperm(N)[: int(N * 0.9)]
    test_patches_90[:, mask_indices, :] = float("nan")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = masked_vit(test_patches_90)

    print("\nTop 15 CUDA operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def quick_size_test():
    """Test different model sizes."""
    device = torch.device("cuda")

    print("=" * 80)
    print("Model Size Comparison")
    print("=" * 80)

    models = ["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224"]

    B = 32
    ratio = 0.9

    for model_name in models:
        try:
            print(f"\n{model_name}:")
            print("-" * 40)

            vit = timm.create_model(model_name, pretrained=False).to(device)
            masked_vit = EfficientMaskedTimmViT(vit).to(device)

            x = torch.randn(B, 3, 224, 224, device=device)

            with torch.no_grad():
                patches = vit.patch_embed(x)

            N = patches.shape[1]

            # Baseline
            time_baseline = profile_forward_pass(masked_vit, patches, num_iter=50)

            # 90% masking
            test_patches = patches.clone()
            mask_indices = torch.randperm(N)[: int(N * ratio)]
            test_patches[:, mask_indices, :] = float("nan")

            time_masked = profile_forward_pass(masked_vit, test_patches, num_iter=50)

            speedup = time_baseline / time_masked

            print(f"  0% masking:   {time_baseline:.2f} ms")
            print(f"  90% masking:  {time_masked:.2f} ms")
            print(f"  Speedup:      {speedup:.2f}x")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PART 1: Detailed Profiling")
    print("=" * 80)
    detailed_profile()

    print("\n\n" + "=" * 80)
    print("PART 2: Model Size Comparison")
    print("=" * 80)
    quick_size_test()
