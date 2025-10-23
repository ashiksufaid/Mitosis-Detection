# Glioma Mitosis Detection (ResNet50 Classifier)

This repository implements a simple **training pipeline** for detecting mitotic cells in glioma histopathology images using a fine-tuned **ResNet-50** model.  
It was built as part of participation in the [Glioma-MCD 2025](https://www.kaggle.com/competitions/glioma-mcd-2025) challenge.

---

## Problem Overview

Gliomas are brain tumors whose aggressiveness is often assessed by identifying mitotic figures — cells undergoing division — in biopsy images.  
Manual annotation is tedious and subjective, so the goal is to train a model to **classify cropped regions of interest (ROIs)** as:

- **Mitosis (1)** – actively dividing cell  
- **Non-Mitosis (0)** – normal cell or background  

The dataset consists of histopathology image patches (with corresponding JSON annotations). Each ROI is labeled as mitotic or non-mitotic.

---

## Repository Structure
Mitosis-Detection/
├── Dataset.py # Custom PyTorch Dataset to load and preprocess ROI crops
├── train.py # Training loop with loss, accuracy tracking, checkpoint saving
├── main.py # Main script to build model, set up data loader, and run training
└── predict.py # (Currently same as train.py – placeholder for inference)

---

## Code Overview

### `Dataset.py`

Defines the `GliomaDataset` class, which:

- Reads `.jpg` images and matching `.json` annotation files from a directory  
- Parses each JSON to locate bounding boxes (`xmin, ymin, xmax, ymax`) around annotated shapes  
- Extracts those ROIs, resizes and pads them to a fixed size (default `48×48`)  
- Applies optional augmentations (flip, rotate, color jitter) when `do_aug=True`  
- Returns `(image_tensor, label_tensor)` for each sample  

Each JSON file must include:
```json
{
  "shapes": [
    {
      "label": "Mitosis",
      "points": [[x1, y1], [x2, y2]]
    }
  ]
}
```

### 'train.py'

Implements the main training loop:
- Moves model and data to the available device
- Fora each epoch:
    - Forward pass → loss computation → backward pass → weight update
    - Tracks and prints batch loss and epoch accuracy
- Saves model weights to
  ```bash
  checkpoints/model_weights.pth
  ```
### `main.py`

- Loads a ResNet-50 backbone pretrained on ImageNet

- Replaces the final fully-connected layer for binary classification

- Freezes early layers (up to layer2) and fine-tunes the deeper blocks (layer3, layer4, fc)

- Creates the GliomaDataset and DataLoader

- Defines:

    - Loss: BCEWithLogitsLoss

    - Optimizer: Adam

    - Epochs: 10

- Calls train_model() from train.py to begin training

### `predict.py`

Used to evaluate the trained model on a test dataset.

Steps performed:

- Loads the trained ResNet-50 checkpoint (model_weights2.pth).

- Builds a test GliomaDataset and DataLoader.

- Runs inference on all test images.

- Computes classification metrics:

    - Accuracy

    - Precision

    - Recall (Sensitivity)

    - Specificity

    - F1-score

Displays a detailed confusion matrix:
```yaml
                    Predicted: Non-mitosis     Predicted: Mitosis
Actual: Non-mitosis        TN                        FP
Actual: Mitosis            FN                        TP
```


