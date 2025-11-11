import os
import shutil
import numpy as np
from PIL import Image, ImageFont, ImageDraw
# MODIFIED: Added gaussian_filter and measure (for marching_cubes)
from scipy.ndimage import rotate, zoom, affine_transform, gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.measure as measure

# --- Configuration ---
FONT_PATH = 'KantumruyPro-Regular.ttf' 
DATASET_DIR = "khmer_numeral_dataset_smooth_64" # 
NUMERALS = "១" # 

SAMPLES_PER_NUMERAL = 500

# VOXEL_SIZE must match IMG_SIZE
IMG_SIZE = 64
VOXEL_SIZE = 64 # MUST match IMG_SIZE for true high-res
EXTRUSION_DEPTH = 16 # Increased depth slightly for 64-grid

def rasterize_char(char, font, size):
    """
    Renders a single character to a 2D ANTI-ALIASED numpy array.
    """
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    try:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        x_pos = (size - char_width) // 2 - bbox[0]
        y_pos = (size - char_height) // 2 - bbox[1]
    except TypeError:
        char_width, char_height = font.getsize(char)
        x_pos = (size - char_width) / 2
        y_pos = (size - char_height) / 2
        
    draw.text((x_pos, y_pos), char, font=font, fill=255)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Added Gaussian filter for 2D anti-aliasing (smooth edges)
    return gaussian_filter(img_array, sigma=0.5)

def extrude_to_3d(img_2d, depth, voxel_size):
    """
    Extrudes the 2D image into a 3D SMOOTH DENSITY CLOUD.
    """
    if img_2d.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    start_depth = (voxel_size - depth) // 2
    
    for i in range(depth):
        z_slice = start_depth + i
        if 0 <= z_slice < voxel_size:
            # Create a smooth "falloff" at the edges
            fade_factor = 1.0 - (abs(i - depth/2) / (depth/2)) * 0.3
            grid[:, :, z_slice] = img_2d * fade_factor
            
    # Apply 3D filter to smooth the whole shape into a density cloud
    return gaussian_filter(grid, sigma=1.0)

def apply_augmentations(voxel_grid):
    """
    Applies random transformations using HIGH-QUALITY interpolation.
    """
    
    # Changed all 'order=0' to 'order=3' (cubic interpolation)
    # This is ESSENTIAL for smooth transformations.

    # Rotation
    angle_x, angle_y, angle_z = np.random.uniform(-15, 15, 3)
    voxel_grid = rotate(voxel_grid, angle_x, axes=(1, 2), reshape=False, mode='constant', cval=0, order=3)
    voxel_grid = rotate(voxel_grid, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=3)
    voxel_grid = rotate(voxel_grid, angle_z, axes=(0, 1), reshape=False, mode='constant', cval=0, order=3)

    # Scaling
    scale_factor = np.random.uniform(0.9, 1.1)
    zoomed = zoom(voxel_grid, scale_factor, mode='constant', cval=0, order=3)
    
    (zx, zy, zz) = zoomed.shape
    (vx, vy, vz) = voxel_grid.shape
    x_start = max(0, (zx - vx) // 2)
    y_start = max(0, (zy - vy) // 2)
    z_start = max(0, (zz - vz) // 2)
    
    cropped_zoomed = zoomed[
        x_start : x_start + vx,
        y_start : y_start + vy,
        z_start : z_start + vz,
    ]
    
    (cx, cy, cz) = cropped_zoomed.shape
    pad_x = vx - cx
    pad_y = vy - cy
    pad_z = vz - cz
    voxel_grid = np.pad(cropped_zoomed, (
        (pad_x//2, pad_x - pad_x//2), 
        (pad_y//2, pad_y - pad_y//2), 
        (pad_z//2, pad_z - pad_z//2)
    ), 'constant')
    
    # Shearing
    shear_val = np.random.uniform(-0.2, 0.2)
    shear_matrix = np.array([[1, shear_val, 0], [0, 1, 0], [0, 0, 1]])
    voxel_grid = affine_transform(voxel_grid, shear_matrix, mode='constant', cval=0, order=3)
    
    return voxel_grid

def generate_dataset():
    """Main function to generate the entire dataset."""
    if not os.path.exists(FONT_PATH):
        print(f"Font file not found at '{FONT_PATH}'. Please update the path.")
        return False

    if os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' already exists. Recreating it.")
        shutil.rmtree(DATASET_DIR)
        
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Created dataset directory: '{DATASET_DIR}'")

    # Changed font size to fill the 64x64 canvas
    font = ImageFont.truetype(FONT_PATH, int(IMG_SIZE * 1.1)) 
    
    total_files = len(NUMERALS) * SAMPLES_PER_NUMERAL
    count = 0
    
    for i, char in enumerate(NUMERALS):
        print(f"\nGenerating data for numeral '{char}' ({i+1}/{len(NUMERALS)})...")
        
        base_2d = rasterize_char(char, font, IMG_SIZE)
        base_3d = extrude_to_3d(base_2d, EXTRUSION_DEPTH, VOXEL_SIZE)
        
        for j in range(SAMPLES_PER_NUMERAL):
            augmented_voxel = apply_augmentations(base_3d.copy())
            
            # The data is now a "density cloud" (floats from 0.0 to 1.0)
            filename = f"numeral_{i}_{j:04d}.npy"
            filepath = os.path.join(DATASET_DIR, filename)
            np.save(filepath, augmented_voxel)
            
            count += 1
            print(f"\r Saved {count}/{total_files}", end="")
    
    print(f"\n\n Dataset generation complete. {total_files} files created.")
    return True

def visualize_sample(dataset_dir):
    """
    MODIFIED: Visualizes the sample using MARCHING CUBES.
    This shows the true smooth surface, not the blocky voxels.
    """
    print("\n Visualizing a random sample (as a smooth surface)...")
    try:
        files = os.listdir(dataset_dir)
        if not files:
            print("No files found in dataset directory.")
            return
        
        sample_file = np.random.choice(files)
        sample_path = os.path.join(dataset_dir, sample_file)
        
        # Load the density grid
        voxel_grid = np.load(sample_path)
        
        # Extract the surface mesh using Marching Cubes
        # The 'level' is the density threshold 
        verts, faces, _, _ = measure.marching_cubes(voxel_grid, level=0.1)
        
        # Plot the 3D mesh
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                        color='cyan', alpha=0.8, shade=True)
        ax.set_title(f"Sample: {sample_file}")
        plt.show()

    except Exception as e:
        print(f"Could not visualize sample. Error: {e}")

if __name__ == "__main__":
    if generate_dataset():
        visualize_sample(DATASET_DIR)
