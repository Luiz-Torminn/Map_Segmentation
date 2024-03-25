#%%
import os
import re
import numpy as np
from patchify import patchify
from PIL import Image
from pathlib import Path

def create_folders():
    FOLDERS = ["train", "validation", "test"]
    
    for folder in FOLDERS:
        if not os.path.exists(f"./{folder}"):
            img_folder = f"{folder}/images"
            msk_folder = f"{folder}/masks"
            
            os.makedirs(img_folder) if not os.path.exists(img_folder) else print("Image folder already exists...")
            os.makedirs(msk_folder) if not os.path.exists(msk_folder) else print("Mask folder already exists...")
            
create_folders()

def create_patches(src, path_save):
    path_split = os.path.split(src)
    tile_num = re.findall(r"\d+", path_split[0])[0]
    image = Image.open(src)
    image = np.asarray(image)

    if len(image.shape) > 2:
        patches = patchify(image, (320,320,3), step=300)
        path_stem = Path(src).stem
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                n = i * patches.shape[1] + j
                patch = Image.fromarray(patch)
                patch.save(f"{path_save}/{path_stem}_tile_{tile_num}_patch_{n}.png") 
       
# %%
for path_name, _, file_name in os.walk("data"):
   for f in sorted(file_name, reverse=True):
       
       if f != "classes.json":
           
           path_split = os.path.split(path_name)
           tile_num = re.findall(r"\d+", path_split[0])[0]
           img_type = path_split[1]
           
           if tile_num == '1':
               image_dir = "validation"
               mask_dir = "validation"
                   
           elif tile_num == '3':
               image_dir = "test"
               mask_dir = "test"
                
           elif tile_num in ['4', '5', '6', '7', '8']:
               image_dir = "train"
               mask_dir= "train"
               
           src = os.path.join(path_name, f)
           path_stem = Path(src).stem
           
           image_file_name = f"{path_split[0]}/images/{path_stem}.jpg" 
           mask_file_name = f"{path_split[0]}/masks/{path_stem}.png" 
           
           if os.path.exists(image_file_name) and os.path.exists(mask_file_name):
             if img_type == "images":
                dest = os.path.join(image_dir, "images")
                create_patches(src, dest)

             if img_type == "masks":
                dest = os.path.join(mask_dir, "masks")
                create_patches(src, dest)
# %%
