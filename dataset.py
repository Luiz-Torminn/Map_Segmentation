#%%
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

#%%
class SemanticDataset(Dataset):
    def __init__(self, image_path, mask_path):
        super(SemanticDataset).__init__()
        
        self.image_names = os.listdir(f"{image_path}")
        self.image_paths = [f"{image_path}/{i}" for i in self.image_names]
        
        self.masks_names = os.listdir(f"{mask_path}")
        self.mask_paths = [f"{mask_path}/{i}" for i in self.masks_names]
        
        # Check if all images have a corresponding mask
        
        self.img_stem = [Path(i).stem for i in sorted(self.image_paths) if Path(i).stem != ".DS_Store"]
        self.mask_stem = [Path(i).stem for i in sorted(self.mask_paths) if Path(i).stem != ".DS_Store"]
        
        self.img_mask_corr = set(self.img_stem) & set(self.mask_stem)
        
        self.image_paths = [i for i in self.image_paths if Path(i).stem in self.img_mask_corr]
        self.mask_paths = [i for i in self.mask_paths if Path(i).stem in self.img_mask_corr]
        
        # Image transformation:
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
        
    
    def mask_converter(self, mask):
        mask[mask == 155] = 0 #unlabeled
        
        mask[mask == 44] = 1 #buildings
        
        mask[mask == 91] = 2 #land
        
        mask[mask == 171] = 3 #water
        
        mask[mask == 172] = 4 #road
        
        mask[mask == 212] = 5 #vegetation
        
        return mask
        
    def __len__(self):
        return len(self.img_mask_corr)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Structure must be: BS, CC, H, W
        image = self.transformation(image) 
        # image = torch.transpose(image, (2, 0, 1))
        #Same thing with the masks
        mask = cv2.imread(self.mask_paths[index], 0)
        mask = self.mask_converter(mask)
    
        return (image, mask)
        