import torch
import os
import json
from Dataset import GliomaDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torchvision.models as models

def collect_roi_info(root_dir):
    images = sorted([f for f in os.listdir(root_dir) if f.endswith("jpg")])
    jsons = sorted([f for f in os.listdir(root_dir) if f.endswith("json")])
    roi_info = []
    for image, js in zip(images, jsons):
        img_path = os.path.join(root_dir, image)
        json_path = os.path.join(root_dir, js)
        with open(json_path, "r") as f:
            data = json.load(f)
            annotations = data["shapes"]
        for shape in annotations:
            roi_label = shape['label']  # e.g., "Blank 1"
            roi_info.append({
                "Image ID": image,
                "Label ID": roi_label
            })
    return roi_info

def main():
    test_dir = '/home/ashiksufaid/summer proj 2/Data_122824/Glioma_MDC_2025_test'
    batch_size = 32

    # 1. Collect ROI info in the same order as Dataset
    roi_info = collect_roi_info(test_dir)

    # 2. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=None)

    num_classes = 1
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(os.getcwd())
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'model_weights_resnet50_ft.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Best epoch:", checkpoint['epoch'])
    print("Best val loss:", checkpoint['val_loss'])

    model.to(device)
    model.eval()

    # 3. Prepare DataLoader
    dataset = GliomaDataset(test_dir, transform="test")
    loader = DataLoader(dataset, batch_size=32)

    # 4. Run inference and collect predictions
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy().flatten()
            all_preds.extend(preds.tolist())

    # 5. Combine ROI info and predictions
    assert len(roi_info) == len(all_preds), "ROI info and predictions length mismatch!"
    df = pd.DataFrame(roi_info)
    df["Prediction"] = all_preds
    df.insert(0, "Row ID", df.index + 1)

    # 6. Save to CSV
    df.to_csv("no_pretrain_resnet50_ft_pred.csv", index=False)
    print("Saved predictions to predictions.csv")

if __name__ == "__main__":
    main()