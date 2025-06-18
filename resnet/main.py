import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from Dataset import GliomaDataset
from torch.utils.data import DataLoader
from train import train_model

resnet = models.resnet50(pretrained=True)

num_classes = 1
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True
for param in resnet.fc.parameters():
    param.requires_grad = True

train_dir = "/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_training"
train_dataset = GliomaDataset(train_dir, do_aug=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle = True)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
resnet.train()
loss = train_model(resnet, epochs, train_loader, criterion, optimizer, device)
