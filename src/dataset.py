import torch
from torch.utils.data import Dataset
import numpy as np
import os

class VoxelDataset(Dataset):
    """
    Dataset for loading 3D voxel grids from .npy files
    
    Args:
        root_dir: Directory containing .npy files
        transform: Optional transform to apply to samples
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all .npy files in the directory
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {root_dir}")
        
        print(f"Found {len(self.files)} .npy file(s) in {root_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load the .npy file
        file_path = os.path.join(self.root_dir, self.files[idx])
        voxel_grid = np.load(file_path)
        
        # Ensure it's float32
        voxel_grid = voxel_grid.astype(np.float32)
        
        # Convert to tensor and add channel dimension
        # Shape: (64, 64, 64) -> (1, 64, 64, 64)
        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0)
        
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
        
        return voxel_tensor
