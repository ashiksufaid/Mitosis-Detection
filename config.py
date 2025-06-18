import torch
from torch import nn

batch_size = 16
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam
