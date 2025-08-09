import math
import torch
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

def _sample_block_size(
    height: int,
    width: int,
    min_scale: float,
    max_scale: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
):
    """Sample a single block mask for an image.

    Args:
        height (int): Height of the image in patches.
        width (int): Width of the image in patches.
        min_scale (float): Minimum scale factor for block area relative to total image area.
        max_scale (float): Maximum scale factor for block area relative to total image area.
        min_aspect_ratio (float): Minimum aspect ratio (height/width) for the block.
        max_aspect_ratio (float): Maximum aspect ratio (height/width) for the block.
    
    Returns:
        tuple[int, int]: A tuple (h, w) containing the sampled block height and width.
    """
    _rand = torch.rand(1).item()
    mask_scale = min_scale + _rand * (max_scale - min_scale)
    max_keep = int(height * width * mask_scale)
    aspect_ratio = min_aspect_ratio + _rand * (max_aspect_ratio - min_aspect_ratio)
    
    # Compute block height and width (given scale and aspect-ratio)
    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    h = min(h, height)

    w = int(round(math.sqrt(max_keep / aspect_ratio)))
    w = min(w, width)

    return (h, w)


def _sample_block_mask(
    image_size: Tuple[int, int],
    block_size: Tuple[int, int],
    min_keep: int = 1,
):
    """Sample a single block mask for an image.
    Because mask positions are chosen randomly, we can occasionally end up with a mask that is too small.
    This function will retry until a valid mask is found.

    Args:
        height (int): Height of the image in patches.
        width (int): Width of the image in patches.
        min_scale (float): Minimum scale factor for block area relative to total image area.
        max_scale (float): Maximum scale factor for block area relative to total image area.
        min_aspect_ratio (float): Minimum aspect ratio (height/width) for the block.
        max_aspect_ratio (float): Maximum aspect ratio (height/width) for the block.
        min_keep (int): Minimum number of patches that must be in the block.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mask: Binary tensor indices of masked patches (flattened)
            - mask_complement: Binary tensor where 1 = available for future blocks
    """
    h, w = block_size
    height, width = image_size
    max_attempts = 20

    for _ in range(max_attempts):
        top = torch.randint(0, height - h + 1, (1,)).item()
        left = torch.randint(0, width - w + 1, (1,)).item()
        
        mask = torch.zeros((height, width), dtype=torch.int32)
        mask[top:top+h, left:left+w] = 1
        # Return the first mask that satisfies min_keep.
        if torch.sum(mask) >= min_keep: return mask

    # If we run out of attempts, return whatever we had last.
    else: return mask


def multi_block_mask(
    height: int,
    width: int,
    num_blocks: int = 4,
    context_scale: Tuple[float, float] = (0.85, 1.0), # -- enc mask scale
    target_scale: Tuple[float, float] = (0.15, 0.2),  # -- pred mask scale
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    min_keep: int = 1,
) -> torch.Tensor:
    """Generate block mask(s) for an image.
    
    Args:
        height: Height in patches
        width: Width in patches
        mask_ratio: Fraction to mask
        num_blocks: Number of mask blocks
        aspect_ratio: (min, max) aspect ratio for blocks
        min_keep: Minimum patches to keep unmasked
        generator: For reproducibility
        
    Returns:
        Binary mask of shape (height, width) where 1 = masked, 0 = visible
    """
    min_scale, max_scale = context_scale
    # No aspect ratio for the context block
    h, w = _sample_block_size(height, width, min_scale, max_scale, 1., 1.)

    # -- Sample context mask
    mask_enc = _sample_block_mask(
        (height, width),
        (h, w),
        min_keep,
    )

    min_scale, max_scale = target_scale
    min_aspect_ratio, max_aspect_ratio = aspect_ratio

    masks_pred = []
    for _ in range(num_blocks):
        h, w = _sample_block_size(
            height, width,
            min_scale, max_scale,
            min_aspect_ratio, max_aspect_ratio
        )
        masks_pred += [
            _sample_block_mask(
            (height, width),
            (h, w),
            min_keep
        )]

    # NOTE Since 1 == discard and 0 == keep, combining masks is an OR operation
    combined_mask = (1 - mask_enc).clone()
    for mask in masks_pred[1:]:
        combined_mask = torch.logical_or(combined_mask, mask)

    # -- Return masks
    return mask_enc, masks_pred, combined_mask

def visualize_masking_strategy(height=14, width=14, num_examples=6, save_path="ijepa_masking_visualization.png"):
    """
    Visualize the I-JEPA masking strategy with multiple examples.
    
    Args:
        height: Image height in patches
        width: Image width in patches  
        num_examples: Number of masking examples to show
        save_path: Path to save the visualization
    """
    
    fig, axes = plt.subplots(2, num_examples, figsize=(3*num_examples, 6))
    if num_examples == 1:
        axes = axes.reshape(2, 1)
    
    # Set random seed for reproducible examples
    torch.manual_seed(42)
    
    for i in range(num_examples):
        # Generate masks
        context_mask, target_masks, combined_mask = multi_block_mask(
            height, width,
            num_blocks=4,
            context_scale=(0.85, 1.0),
            target_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5)
        )
        
        # Convert to numpy for visualization
        context_np = context_mask.numpy()
        combined_np = combined_mask.numpy()
        
        # Create visualization grids
        vis_grid_separate = np.zeros((height, width))
        vis_grid_combined = np.zeros((height, width))
        
        # Color coding: 0=visible, 1=context, 2-5=targets 1-4
        # For separate visualization
        vis_grid_separate[context_np == 1] = 1  # Context in blue
        for j, target_mask in enumerate(target_masks):
            target_np = target_mask.numpy()
            vis_grid_separate[target_np == 1] = j + 2  # Targets in different colors
        
        # For combined visualization  
        vis_grid_combined[combined_np == 1] = 1  # All masked regions
        
        # Plot separate masks (context + targets)
        ax1 = axes[0, i]
        colors = ['white', 'lightblue', 'red', 'orange', 'green', 'purple']
        cmap = ListedColormap(colors[:6])
        im1 = ax1.imshow(vis_grid_separate, cmap=cmap, vmin=0, vmax=5)
        ax1.set_title(f'Example {i+1}: Context + Targets', fontsize=10)
        ax1.set_xticks(range(0, width, 2))
        ax1.set_yticks(range(0, height, 2))
        ax1.grid(True, alpha=0.3)
        
        # Add grid lines for patches
        for x in range(width + 1):
            ax1.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for y in range(height + 1):
            ax1.axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # Plot combined mask
        ax2 = axes[1, i]
        cmap_combined = ListedColormap(['white', 'black'])
        im2 = ax2.imshow(vis_grid_combined, cmap=cmap_combined, vmin=0, vmax=1)
        ax2.set_title(f'Combined Mask', fontsize=10)
        ax2.set_xticks(range(0, width, 2))
        ax2.set_yticks(range(0, height, 2))
        ax2.grid(True, alpha=0.3)
        
        # Add grid lines for patches
        for x in range(width + 1):
            ax2.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for y in range(height + 1):
            ax2.axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='white', label='Visible patches'),
        patches.Patch(color='lightblue', label='Context block'),
        patches.Patch(color='red', label='Target block 1'),
        patches.Patch(color='orange', label='Target block 2'),
        patches.Patch(color='green', label='Target block 3'),
        patches.Patch(color='purple', label='Target block 4')
    ]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=10)
    
    plt.suptitle('I-JEPA Masking Strategy Visualization\n(Top: Context + Target blocks, Bottom: Combined mask)', 
                 fontsize=14, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

def analyze_masking_statistics(height=14, width=14, num_samples=1000):
    """
    Analyze statistics of the masking strategy.
    """
    context_scales = []
    target_scales = []
    aspect_ratios = []
    
    torch.manual_seed(42)
    
    for _ in range(num_samples):
        context_mask, target_masks, combined_mask = multi_block_mask(
            height, width,
            num_blocks=4,
            context_scale=(0.85, 1.0),
            target_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5)
        )
        
        total_patches = height * width
        context_scale = torch.sum(context_mask).item() / total_patches
        context_scales.append(context_scale)
        
        for target_mask in target_masks:
            target_scale = torch.sum(target_mask).item() / total_patches
            target_scales.append(target_scale)
            
            # Calculate aspect ratio of target
            coords = torch.where(target_mask == 1)
            if len(coords[0]) > 0:
                h_extent = coords[0].max() - coords[0].min() + 1
                w_extent = coords[1].max() - coords[1].min() + 1
                aspect_ratios.append(h_extent.item() / w_extent.item())
    
    # Create statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(context_scales, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0].set_title('Context Block Scales')
    axes[0].set_xlabel('Scale (fraction of image)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(context_scales), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(context_scales):.3f}')
    axes[0].legend()
    
    axes[1].hist(target_scales, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Target Block Scales')
    axes[1].set_xlabel('Scale (fraction of image)')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(target_scales), color='red', linestyle='--',
                   label=f'Mean: {np.mean(target_scales):.3f}')
    axes[1].legend()
    
    axes[2].hist(aspect_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[2].set_title('Target Block Aspect Ratios')
    axes[2].set_xlabel('Aspect Ratio (height/width)')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(np.mean(aspect_ratios), color='red', linestyle='--',
                   label=f'Mean: {np.mean(aspect_ratios):.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('ijepa_masking_statistics.png', dpi=300, bbox_inches='tight')
    print("Statistics saved to: ijepa_masking_statistics.png")
    plt.close()
    
    return {
        'context_scales': context_scales,
        'target_scales': target_scales, 
        'aspect_ratios': aspect_ratios
    }

if __name__ == "__main__":
    print("Generating I-JEPA masking visualizations...")
    
    # Create main visualization
    visualize_masking_strategy(height=14, width=14, num_examples=8, 
                             save_path="ijepa_masking_examples.png")
    
    # Generate statistics
    stats = analyze_masking_statistics()
    
    print(f"\nMasking Statistics:")
    print(f"Context scale - Mean: {np.mean(stats['context_scales']):.3f}, Std: {np.std(stats['context_scales']):.3f}")
    print(f"Target scale - Mean: {np.mean(stats['target_scales']):.3f}, Std: {np.std(stats['target_scales']):.3f}")
    print(f"Aspect ratio - Mean: {np.mean(stats['aspect_ratios']):.3f}, Std: {np.std(stats['aspect_ratios']):.3f}")
    
    print("\nVisualization complete! Check the generated PNG files.")