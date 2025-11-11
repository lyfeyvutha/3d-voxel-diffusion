import torch
import numpy as np
import os
import argparse
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.measure as measure
from tqdm import tqdm

# Import our custom files
from src.config import *
from src.model import UNet3D
from src.utils import get_ddpm_schedule, extract, p_sample
from src.dataset import VoxelDataset

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Improved Plotting Functions ---
def plot_smooth_surface(ax, grid, title=''):
    """Extracts and plots a smooth mesh from a density grid with diagnostics."""
    
    # Print diagnostics
    print(f"\n{title} Statistics:")
    print(f"  Shape: {grid.shape}")
    print(f"  Min: {grid.min():.4f}, Max: {grid.max():.4f}")
    print(f"  Mean: {grid.mean():.4f}, Std: {grid.std():.4f}")
    print(f"  Non-zero voxels: {(grid > 0.01).sum()} / {grid.size}")
    
    try:
        # Try multiple threshold levels
        for level in [0.5, 0.3, 0.1, 0.05, 0.01]:
            try:
                verts, faces, _, _ = measure.marching_cubes(grid, level=level)
                if len(verts) > 0:
                    print(f"  ✓ Surface extracted at level {level} with {len(verts)} vertices")
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                    color='magenta', alpha=0.8, shade=True)
                    ax.set_title(f"{title}\n(level={level})", fontsize=10)
                    ax.set_xlim(0, grid.shape[0])
                    ax.set_ylim(0, grid.shape[1])
                    ax.set_zlim(0, grid.shape[2])
                    return
            except (ValueError, RuntimeError) as e:
                continue
        
        # If all levels failed, show a voxel visualization instead
        print(f"  ✗ No surface found at any level, showing voxel visualization")
        plot_voxel_fallback(ax, grid, title)
        
    except Exception as e:
        ax.set_title(f"{title}\n(Error: {str(e)[:50]})", fontsize=8)
        print(f"  ✗ Plotting Error: {e}")

def plot_voxel_fallback(ax, grid, title):
    """Fallback visualization using voxel scatter plot."""
    threshold = np.percentile(grid, 95)
    coords = np.where(grid > threshold)
    
    if len(coords[0]) == 0:
        ax.text(32, 32, 32, "Empty\nVolume", ha='center', va='center', fontsize=12)
        ax.set_title(f"{title}\n(No density)", fontsize=10)
        return
    
    n_points = min(len(coords[0]), 2000)
    indices = np.random.choice(len(coords[0]), n_points, replace=False)
    
    x = coords[0][indices]
    y = coords[1][indices]
    z = coords[2][indices]
    colors = grid[coords[0][indices], coords[1][indices], coords[2][indices]]
    
    scatter = ax.scatter(x, y, z, c=colors, cmap='magma', s=5, alpha=0.6)
    ax.set_title(f"{title}\n(voxel view, threshold={threshold:.3f})", fontsize=9)
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_zlim(0, grid.shape[2])

def plot_comparison_with_slices(original_grid, generated_grid, save_path):
    """Create a comprehensive comparison plot with 3D views and 2D slices."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: 3D surface plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_smooth_surface(ax1, original_grid, title="Original Sample (3D)")
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_smooth_surface(ax2, generated_grid, title="Generated Sample (3D)")
    
    # Difference plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    diff = np.abs(original_grid - generated_grid)
    plot_smooth_surface(ax3, diff, title="Absolute Difference")
    
    # Row 2: 2D center slices
    mid = original_grid.shape[0] // 2
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(original_grid[mid, :, :], cmap='viridis', vmin=0, vmax=1)
    ax4.set_title("Original (Center Slice)")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(generated_grid[mid, :, :], cmap='viridis', vmin=0, vmax=1)
    ax5.set_title("Generated (Center Slice)")
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(diff[mid, :, :], cmap='hot', vmin=0, vmax=1)
    ax6.set_title("Difference (Center Slice)")
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved comprehensive plot to: {save_path}")

# --- Fast Sampling Loop ---
@torch.no_grad()
def p_sample_loop_fast(model, shape, full_timesteps, sampling_timesteps, schedule):
    """
    DDIM-style fast sampler.

    Args:
        model: trained U-Net.
        shape: output tensor shape.
        full_timesteps: total diffusion steps used during training.
        sampling_timesteps: number of steps to use at inference.
        schedule: precomputed diffusion coefficients.
    """
    print(f"\n--- Starting {sampling_timesteps}-Step Fast Inference ---")
    model.eval()

    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)

    intermediate_steps = []
    ts = torch.linspace(full_timesteps - 1, 0, sampling_timesteps, device=device).long()
    alphas_cumprod = schedule["alphas_cumprod"]

    for i in tqdm(range(sampling_timesteps), desc="Sampling", total=sampling_timesteps):
        t = ts[i].unsqueeze(0)
        eps_theta = model(img, t)

        alpha_bar_t = extract(alphas_cumprod, t, img.shape)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))

        x0_pred = (img - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if i == sampling_timesteps - 1:
            img = x0_pred
        else:
            t_next = ts[i + 1].unsqueeze(0)
            alpha_bar_next = extract(alphas_cumprod, t_next, img.shape)
            sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
            sqrt_one_minus_alpha_bar_next = torch.sqrt(torch.clamp(1.0 - alpha_bar_next, min=1e-12))

            img = (sqrt_alpha_bar_next * x0_pred) + (sqrt_one_minus_alpha_bar_next * eps_theta)

        img_denoised = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        grid_3d = img_denoised.squeeze().cpu().numpy()
        slice_2d = grid_3d[grid_3d.shape[0] // 2, :, :]
        intermediate_steps.append((slice_2d * 255).astype(np.uint8))

    model.train()
    print("--- Inference complete ---")

    return grid_3d, intermediate_steps


@torch.no_grad()
def p_sample_loop_ddpm(model, shape, timesteps, schedule):
    """Full DDPM sampler that mirrors the training transition kernels."""
    print(f"\n--- Starting {timesteps}-Step DDPM Inference ---")
    model.eval()

    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)
    intermediate_steps = []

    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, schedule)

        img_denoised = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        grid_3d = img_denoised.squeeze().cpu().numpy()
        slice_2d = grid_3d[grid_3d.shape[0] // 2, :, :]
        intermediate_steps.append((slice_2d * 255).astype(np.uint8))

    intermediate_steps.reverse()  # earliest step first for GIF timelines

    model.train()
    print("--- Inference complete ---")

    return grid_3d, intermediate_steps

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained 3D DDPM.")
    parser.add_argument(
        '--model_filename', 
        type=str, 
        default="best_model.pth", 
        help='Filename of the trained model in checkpoints folder.'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help="Number of samples to generate."
    )
    parser.add_argument(
        '--sampler',
        type=str,
        choices=['ddim', 'ddpm'],
        default='ddim',
        help="Sampling strategy: 'ddim' (fast, strided) or 'ddpm' (full T steps)."
    )
    parser.add_argument(
        '--autoencoder_only',
        action='store_true',
        help='Use autoencoder-only weights and skip diffusion sampling.'
    )
    args = parser.parse_args()
    
    # Load model
    if args.autoencoder_only:
        model_path = os.path.join(CHECKPOINT_DIR, "autoencoder_best_model.pth")
    else:
    model_path = os.path.join(CHECKPOINT_DIR, args.model_filename)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print("Loading building blocks...")
    schedule = get_ddpm_schedule(TIMESTEPS, device=device)
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model: {model_path}")
    
    # Load original sample
    try:
        dataset = VoxelDataset(root_dir=DATA_DIR)
        original_grid_tensor = dataset[0]
        
        # Normalize for comparison
        original_grid_normalized = (original_grid_tensor.squeeze().numpy() * 2.0) - 1.0
        original_grid_plot = (original_grid_normalized + 1.0) / 2.0
        
        print(f"Loaded original sample from: {DATA_DIR}")
    except Exception as e:
        print(f"Error loading original sample: {e}")
        return
    
    # Generate samples
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    for i in range(args.count):
        print(f"\n--- Generating Sample {i+1} of {args.count} ---")
        sample_num = i + 1
        shape = (1, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
        
        if args.autoencoder_only:
            model.eval()
            with torch.no_grad():
                input_tensor = (original_grid_tensor * 2.0 - 1.0).unsqueeze(0).to(device)
                t_zero = torch.zeros((1,), device=device, dtype=torch.long)
                predicted_clean = model(input_tensor, t_zero).squeeze().cpu().numpy()
            generated_grid = np.clip((predicted_clean + 1.0) / 2.0, 0.0, 1.0)
            gif_frames = []
            model.train()
        else:
        # Generate
        use_full_ddpm = args.sampler.lower() == 'ddpm'
        if use_full_ddpm:
            generated_grid, gif_frames = p_sample_loop_ddpm(
                model,
                shape,
                TIMESTEPS,
                schedule
            )
        else:
            generated_grid, gif_frames = p_sample_loop_fast(
                model,
                shape,
                TIMESTEPS,
                SAMPLING_TIMESTEPS,
                schedule
            )

        # Compute reconstruction error against original sample
        generated_grid_clamped = np.clip(generated_grid, 0.0, 1.0)
        original_grid_clamped = np.clip(original_grid_plot, 0.0, 1.0)
        l2_error = np.sqrt(np.mean((generated_grid_clamped - original_grid_clamped) ** 2))
        print(f"L2 error vs original: {l2_error:.6f}")
        
        # Save GIF
        gif_filename = f"denoising_process_{args.model_filename.split('.')[0]}_sample_{sample_num}.gif"
        gif_path = os.path.join(SAMPLE_DIR, gif_filename)
        if gif_frames:
        print(f"Saving denoising GIF to: {gif_path}...")
        duration_ms = max(15000 / len(gif_frames), 33)
        imageio.mimsave(gif_path, gif_frames, duration=duration_ms, loop=0)
        else:
            print("Skipping GIF generation (no diffusion frames).")
        
        # Save comparison plot
        plot_filename = f"comparison_plot_{args.model_filename.split('.')[0]}_sample_{sample_num}.png"
        plot_path = os.path.join(SAMPLE_DIR, plot_filename)
        
        print(f"Saving comprehensive comparison plot...")
        generated_grid_plot = np.clip(generated_grid, 0.0, 1.0)
        plot_comparison_with_slices(original_grid_plot, generated_grid_plot, plot_path)
    
    print("\n--- All jobs complete! ---")
    print(f"You can now download the files from '{SAMPLE_DIR}' on OSCAR.")

if __name__ == "__main__":
    main()
