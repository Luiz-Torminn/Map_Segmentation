#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SemanticDataset
import segmentation_models_pytorch as smp 
import matplotlib.pyplot as plt
from train import*
from dataset import*
from graphs import*

#%%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOAD_MODEL = True
IOU = True

tstdata = SemanticDataset(image_path = "path_to_image_file", mask_path = "path_to_image_masks")
test_loader = DataLoader(tstdata, batch_size = 1, shuffle = True)

#%%
MODEL = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet", 
    classes=6,
    activation="sigmoid"
)

model = MODEL.to(DEVICE)
loss = smp.losses.JaccardLoss("multiclass", 6, from_logits = True)

avaluation_parameters = {
    "Accuracy": [],
    "IoU Score": [],
    }
# %%
def model_loader(model, path = f"{os.getcwd()}saves/model/model_checkpoint.pth.tar"):
    model.load_state_dict(torch.load(path)["state_dict"])

# %%
if LOAD_MODEL:
    try:
        model_loader(model, "saves/model/model_checkpoint_50_BCELoss.pth.tar")
    
    except FileNotFoundError:
        print("No saved file was found...")
        
# %%
if IOU:
    with torch.no_grad():  
        for image, mask in test_loader:
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            true = mask.float()
            
            output = model(image.float())
            output = output.to(DEVICE)
            
            _, prediction = torch.max(output, 1)
            correct_pixels = (prediction == true).sum().item()
            total_pixels = true.size(0) * true.size(1) * true.size(2)
            
            tp, fp, fn, tn = smp.metrics.get_stats(prediction, mask, mode='multilabel', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            
            avaluation_parameters["IoU Score"].append(iou_score)            
            avaluation_parameters["Accuracy"].append(correct_pixels/total_pixels)
    
    print(f"""
          IoU Score: {(sum(avaluation_parameters['IoU Score'])/len(avaluation_parameters['IoU Score'])) * 100:.2f}%
          Average accuracy: {(sum(avaluation_parameters['Accuracy'])/len(avaluation_parameters['Accuracy'])) * 100:.2f}%
          """)
#  Average IoU: {(sum(avaluation_parameters['IoU Score'])/len(avaluation_parameters['IoU Score'])) * 100}
# %%
img, mask = next(iter(test_loader))
img, mask = img.float().to(DEVICE), mask.float().to(DEVICE)

output = model(img)
prediction = torch.max(output, 1)[1]
prediction = torch.permute(prediction, (1,2,0))


prediction = prediction.cpu().numpy().squeeze()
img = img.cpu().numpy().squeeze()
img = np.transpose(img, (1,2,0))
mask = mask.cpu().numpy().squeeze()

plot_images(img, mask, prediction, f"saves/outputs/prediction.png")
# %%