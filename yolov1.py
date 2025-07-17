import torch.nn as nn 
import torchvision.models as models

#resnet = models.resnet50(weights=None)
#start with non ft resnet

class YoloV1(nn.Module):
    def __init__(self, S=8, B=2, C=2):
        super(YoloV1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, S*S*(B*5 + C))
        )

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1, self.S, self.S, self.B * 5 + self.C)
        return out