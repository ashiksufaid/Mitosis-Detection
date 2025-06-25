import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from Dataset import GliomaDataset
from torch.utils.data import DataLoader
from train import train_model
import matplotlib.pyplot as plt
resnet = models.resnet18(pretrained=False)
print(f"training {resnet}")
num_classes = 1
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

for param in resnet.parameters():
    param.requires_grad = True

train_dir = "/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_training"
train_dataset = GliomaDataset(train_dir, do_aug=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)

val_dir = "/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_tester"
val_dataset = GliomaDataset(val_dir, do_aug=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam
epochs = 40
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
resnet.train()
train_loss, train_acc, val_loss, val_acc = train_model(resnet, epochs, train_loader, val_loader, criterion, optimizer, device)

def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, epochs, save_path="training_results.png"):
    """
    Plots training/validation loss and accuracy side by side.

    Args:
        train_loss (list or np.ndarray): Training loss per epoch.
        val_loss (list or np.ndarray): Validation loss per epoch.
        train_acc (list or np.ndarray): Training accuracy per epoch.
        val_acc (list or np.ndarray): Validation accuracy per epoch.
        epochs (int): Number of epochs.
        save_path (str): Path to save the plot image.
    """
    epochs_range = np.arange(epochs)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    axs[0].plot(epochs_range, train_loss, 'r--', label='Train Loss')
    axs[0].plot(epochs_range, val_loss, 'g-', label='Val Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy subplot
    axs[1].plot(epochs_range, train_acc, 'b--', label='Train Acc')
    axs[1].plot(epochs_range, val_acc, 'm-', label='Val Acc')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, epochs, save_path="training_result_resnet18.png")

