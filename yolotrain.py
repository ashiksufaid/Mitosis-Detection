#Train loop for yolo
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer, device, save_path):
    model = model.to(device)
    best_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for imgs, targets in pbar:
            #move images and targets to device
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #accumulate loss
            running_loss += loss.item() *imgs.size(0) #multiply the loss with number of images in the batch
            pbar.set_postfix(train_loss = loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch Training Loss: {epoch_loss:.4f}")

        #Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    preds = model(imgs)
                    loss = loss_fn(preds, targets)
                    val_loss += loss.item() * imgs.size(0)
            epoch_loss = val_loss / len(val_loader.dataset)
            print(f"Val loss: {epoch_loss}")
        else:
            val_loss = epoch_loss
        val_losses.append(val_loss)
        #Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch" : epoch + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : best_loss
            }, save_path)
            print(f"   → Saved new best model (loss {best_loss:.4f}) to '{save_path}'\n")       
    print("trainign_complete")
    return train_losses, val_losses
