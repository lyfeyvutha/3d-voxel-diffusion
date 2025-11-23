import torch
from torch.utils.data import Dataset
import numpy as np
import os

class VoxelDataset(Dataset):
    """
    Dataset for loading 3D voxel grids from .npy files
    
    Args:
        root_dir: Directory or list of directories containing .npy files
        transform: Optional transform to apply to samples
    """
    def __init__(self, root_dir, transform=None):
        if isinstance(root_dir, (list, tuple)):
            self.root_dirs = list(root_dir)
        else:
            self.root_dirs = [root_dir]
        self.transform = transform
        
        self.files = []
        for directory in self.root_dirs:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
            for current_root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith('.npy'):
                        self.files.append(os.path.join(current_root, filename))
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.root_dirs}")
        
        self.files.sort()
        total = len(self.files)
        print(f"Found {total} .npy file(s) in {', '.join(self.root_dirs)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        voxel_grid = np.load(file_path).astype(np.float32)

        voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0)
        
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
        
        return voxel_tensor
