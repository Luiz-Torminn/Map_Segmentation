#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SemanticDataset
import segmentation_models_pytorch as smp 
import graphs
from train import*
from dataset import*
from earlystopping import EarlyStopping

#%%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# %%
#Hyperparameters
EPOCHS = 20
BATCH_SIZE = 3
LEARNING_RATE = 0.001
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, cooldown=2, verbose=True )

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
es = EarlyStopping(patience = 1,threshold=0.1)

# %%
train = True

while train:
    for epoch in range(EPOCHS):
        train_dataset = SemanticDataset(f"/Users/luizfelipe/Desktop/Python/MachineLearning/Projects/SemanticSegmentation/train_images", f"/Users/luizfelipe/Desktop/Python/MachineLearning/Projects/SemanticSegmentation/train_masks")
        tdata = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = SemanticDataset(f"/Users/luizfelipe/Desktop/Python/MachineLearning/Projects/SemanticSegmentation/val_images", "/Users/luizfelipe/Desktop/Python/MachineLearning/Projects/SemanticSegmentation/val_masks")
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

        scheduler.step(val_loss)
        early = es(model=model, current_loss=val_loss)


        print(f"""\nFor epoch[{epoch + 1}/{EPOCHS}]:
              Train Loss: {train_loss:.4f}
              Validation Loss: {val_loss:.4f}
              Validation Accuracy: {val_accuracy:.2f}%\n
              """)
        
        if not early:
            train = False
            
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            model_saver(checkpoint=checkpoint, path=f"{os.getcwd()}/saves/model/model_checkpoint_{EPOCHS}_BCELoss.pth.tar")
            
            graphs.loss_graph(loss_hist, f"{os.getcwd()}/saves/outputs")
            
            break
        


# %%
