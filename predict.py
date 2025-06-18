import torch
import os
from Dataset import GliomaDataset
from torch.utils.data import DataLoader
from model import Model
from config import device
import numpy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

model = Model()

checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'model_weights.pth')

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

test_dir = '/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_tester'
data = GliomaDataset(test_dir, do_aug=False)
test_loader = DataLoader(data, batch_size=32)

predictions = []
labels = []

with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        predictions.extend(preds.cpu().numpy())
        labels.extend(label.cpu().numpy())

predictions = [float(p[0]) if hasattr(p, '__getitem__') else float(p) for p in predictions]
labels = [float(l) for l in labels]

print(predictions[:10])
print(labels[0:10])
print(len(predictions), len(labels))
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix and Metrics
cm = confusion_matrix(labels, predictions)
print("\nConfusion Matrix:")
print(f"{'':<20}{'Predicted: Non-mitosis':<25}{'Predicted: Mitosis'}")
print(f"{'Actual: Non-mitosis':<20}{cm[0,0]:<25}{cm[0,1]}")
print(f"{'Actual: Mitosis':<20}{cm[1,0]:<25}{cm[1,1]}")

precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
fscore = f1_score(labels, predictions)
# Sensitivity = recall for class 1
sensitivity = recall
# Specificity = recall for class 0
specificity = recall_score(labels, predictions, pos_label=0)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-score: {fscore:.2f}")

