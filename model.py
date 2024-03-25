#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SemanticDataset
import segmentation_models_pytorch as smp 
import graphs
from train import*
from dataset import*

#%%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# %%
#Hyperparameters
EPOCHS = 50
BATCH_SIZE = 3
LEARNING_RATE = 0.0001
LOAD_MODEL = True       

# %%
MODEL = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet", 
    classes=6,
    activation="sigmoid"
)

model = MODEL.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss = torch.nn.CrossEntropyLoss()

loss_hist = {
    "Train Loss": [],
    "Validation Loss": [],
}

#%%
def model_saver(checkpoint, path):
    torch.save(checkpoint, path)
    
def model_loader(model, path):
    model.load_state_dict(torch.load(path)["state_dict"], strict = False)
    model.load_state_dict(torch.load(path)["optimizer"], strict = False)

if LOAD_MODEL:
    try:
        model_loader(model=model, path=f"{os.getcwd()}/saves/model/model_checkpoint.pth.tar")
    except FileNotFoundError:
        print("\nNo checkpoint was found...\n")
# %%
for epoch in range(EPOCHS):
    train_dataset = SemanticDataset(f"path_to_train_images", f"path_to_train_masks")
    tdata = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = SemanticDataset(f"path_to_val_images", f"path_to_val_masks")
    vdata = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    train_loss = train_model(
        dataloader = tdata, 
        device = DEVICE,
        model = model, 
        loss = loss, 
        optimizer = optimizer, 
        epochs = EPOCHS, 
        current_epoch = epoch, 
        batch_size = BATCH_SIZE
    )
    
    val_loss, val_accuracy = val_model(
        dataloader = vdata, 
        device = DEVICE,
        model = model, 
        loss = loss
    )
    
    loss_hist["Train Loss"].append(train_loss)
    loss_hist["Validation Loss"].append(val_loss)
    
    
    print(f"""\nFor epoch[{epoch + 1}/{EPOCHS}]:
          Train Loss: {train_loss:.4f}
          Validation Loss: {val_loss:.4f}
          Validation Accuracy: {val_accuracy:.2f}%\n
          """)
    
checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
model_saver(checkpoint=checkpoint, path=f"{os.getcwd()}/saves/model/model_checkpoint_{EPOCHS}_BCELoss.pth.tar")

#%%
graphs.loss_graph(loss_hist, f"{os.getcwd()}/saves/outputs")
# %%
