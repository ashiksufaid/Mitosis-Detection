import torch
import os
from Dataset import GliomaDataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#calling the model
model = models.resnet50(weights=None)
num_classes = 1
model.fc = nn.Linear(model.fc.in_features, num_classes)
print(os.getcwd())
#making the model using saved weights
checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'model_weights_resnet50_ft.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)
wts = checkpoint["model_state_dict"]
model.load_state_dict(wts)
print(f"Best Epoch: {checkpoint["epoch"]}")
print(f"Best loss: {checkpoint["val_loss"]}")
model.to(device)
model.eval()

test_dir = '/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_tester'
data = GliomaDataset(test_dir, transform='test')
test_loader = DataLoader(data, batch_size=32, shuffle=False)

predictions = []
labels = []
with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        preds = preds.cpu().numpy()
        predictions.extend(preds)
        label = label.cpu().numpy()
        labels.extend(label)

predictions = [float(p[0]) if hasattr(p, '__getitem__') else float(p) for p in predictions]
labels = [float(l) for l in labels]

dat = GliomaDataset(test_dir, aspect_change= False, transform = None)

wrong_dir = "wrong_plots_raw"
os.makedirs(wrong_dir, exist_ok=True)
for idx, (pred,label) in enumerate(zip(predictions, labels)):
    if pred != label:
        crop, _ = dat[idx]
        img_np = np.array(crop)
        plt.imshow(img_np)
        plt.title(f"Pred: {pred}, True label: {label}")
        plt.axis("off")
        plt.savefig(os.path.join(wrong_dir, f"wrong_{idx}_pred{pred}_true{label}.png"))
        plt.close()


