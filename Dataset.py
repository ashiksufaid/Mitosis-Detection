#cropper
#The cropped ROIs should have info weather it is mitotic or non mitotic

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import json
import numpy as np 
from PIL import Image, ImageOps

class GliomaDataset(Dataset):
    def __init__(self, root_dir, target_size=(48,48), aspect_change = True, transform = None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.aspect_change = aspect_change
        self.transform = transform
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith("jpg")])
        self.jsons = sorted([f for f in os.listdir(root_dir) if f.endswith("json")])

        self.samples = []
        for image, js in zip(self.images, self.jsons):
            img_path = os.path.join(root_dir, image)
            json_path = os.path.join(root_dir, js)
            with open(json_path, "r") as f:
                data = json.load(f)
                annotations = data["shapes"]
        
            for shape in annotations:
                points = shape['points']
                xmin, ymin = np.min(points, axis = 0)
                xmax, ymax = np.max(points, axis = 0)
                bbox = [xmin, ymin, xmax, ymax] 
                label = shape['label']
                label = 1 if label == "Mitosis" else 0
                self.samples.append((img_path, bbox, label))

    def __len__(self):
        return len(self.samples)

    def aspect_ratio_resize_and_pad(self,img):
        img = img.copy()
        img.thumbnail(self.target_size, Image.BILINEAR)
        delta_w = self.target_size[0] - img.width
        delta_h = self.target_size[1] - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding, fill=(0, 0, 0))
        return img

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        xmin, ymin, xmax, ymax = map(int, bbox)
        crop = image.crop((xmin, ymin, xmax, ymax))  # PIL crop: (left, upper, right, lower)
        if self.aspect_change:    
            crop = self.aspect_ratio_resize_and_pad(crop)
        if self.transform == 'test':
            crop = self.test_transform(crop)
        elif self.transform == 'train':
            crop = self.train_transform(crop)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return crop, label_tensor
        