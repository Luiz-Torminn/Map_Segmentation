#%%
import torch
import math

def train_model(dataloader, device, model, loss, optimizer, epochs, current_epoch, batch_size):
    model.train()
    train_loss = 0.0
    
    for i, (image, mask) in enumerate(dataloader):
        
        image, mask = image.to(device), mask.to(device)
        
        prediction = model(image.float())

        loss_func = loss(prediction.float(), mask.long())
        loss_func.backward()

        optimizer.step()

        optimizer.zero_grad()
        
        train_loss += loss_func.item()
        
        if (i + 1) % 3 == 0:
            print(f"For Epoch[{current_epoch + 1}/{epochs}] --> Train loss[{i+1}/{len(dataloader)}] = {train_loss/len(dataloader):.4f}")
    
    accumulated_loss = (train_loss/len(dataloader))
        
    return accumulated_loss

def val_model(dataloader, device, model, loss):
    model.eval()
    total_samples = 0
    correct_samples = 0
    eval_loss = 0.0
    
    with torch.no_grad():
        for i, (image, masks) in enumerate(dataloader):
            image, masks = image.to(device), masks.to(device)

            output = model(image.float())
            scores = loss(output.float(), masks.long())

            eval_loss += scores.item()

            # Accuracy
            _, prediction =  output.max(1)
            total_samples += masks.shape[0] * masks.shape[1] * masks.shape[2] #<-- check afterwards
            correct_samples += (prediction == masks).sum()

    accumulated_loss = (eval_loss/len(dataloader))
    val_accuracy = 100 * (correct_samples/total_samples)
    
    return (accumulated_loss, val_accuracy)
        
    
    
# %%
