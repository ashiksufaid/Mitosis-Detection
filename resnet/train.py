import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    model = model.to(device)
    optimizer = optimizer(model.parameters())
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.view(-1,1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) 
            #loss.item() is average loss per sample in a batch. Multiplying it by the batchsize will give the actual loss
            progress_bar.set_postfix(loss=loss.item())
            #To monitor batchwise loss value in real time

            #accuracy calculation for the batch
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        epoch_accuracy = 100 * (correct/total) if total > 0 else 0
        epoch_accuracies.append(epoch_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_path = os.path.join(checkpoints_dir, "model_weights.pth")
    torch.save(model.state_dict(), checkpoint_path)
    return epoch_losses

