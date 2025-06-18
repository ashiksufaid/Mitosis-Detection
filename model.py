import torch.nn.functional as F 
from torch import nn

class Model(nn.Module):
    def __init__(self, in_channels = 3, n_classes = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 128), #fc with 128 neurons
            nn.ReLU(),
            nn.Linear(128,1) #Single output logit
        )

    def forward(self,x):
            x = self.conv(x)
            x = self.fc(x)
            return x