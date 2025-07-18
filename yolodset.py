import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import os
import json
import numpy as np 
from PIL import Image, ImageOps

class YoloV1Dataset(Dataset):
    def __init__(self, root_dir, label_map, S=8, B=2, C=2, img_size = (512, 512), transforms=None):
        self.root_dir = root_dir
        self.label_map = label_map
        self.S = S
        self.B = B
        self.C = C or len(label_map)
        self.img_size = img_size
        self.transforms = transforms or T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

        #open images 
        images = sorted(i for i in os.listdir(root_dir) if i.lower().endswith('.jpg'))

        #now form a list of tuples with these images and corresponding jsons
        self.samples = []

        for image in images: 
            base = os.path.splitext(image)[0]
            json_path = os.path.join(root_dir, base + '.json')
            img_path = os.path.join(root_dir, image)
            if os.path.isfile(json_path):
                self.samples.append((img_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        #Load json 
        with open(json_path, "r") as f:
            data = json.load(f)
        #Has to fix the descripency between json and image
        annotations = data["shapes"]
        json_height = data["imageHeight"]
        json_width = data["imageWidth"]
        
        #parse and normalize boxes (x,y,w,h)
        boxes = []
        labels = []
        for shape in annotations:
            points = shape["points"]
            scale_x = orig_w / json_width
            scale_y = orig_h / json_height
            points = [(x * scale_x, y * scale_y) for x,y in points]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)
        
            # center + size
            xc = (xmin + xmax) / 2 / orig_w
            yc = (ymin + ymax) / 2 / orig_h
            bw = (xmax - xmin) / orig_w
            bh = (ymax - ymin) / orig_h
            boxes.append([xc, yc, bw, bh])
            label = shape["label"]
            label = self.label_map[shape["label"]]
            labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        #build an empty target grid (S * S * (B*5 + C))
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        #fill grid. Remember the grids without images will be full of zeros. 
        #Hence we only need to fill the grid with the objects

        for b in range(boxes.shape[0]):
            x, y, w, h = boxes[b]
            cell_x = int(x * self.S)
            cell_y = int(y * self.S)
            # Clamping to ensure it doesn't go out of bounds
            cell_x = min(self.S - 1, max(0, int(x * self.S)))
            cell_y = min(self.S - 1, max(0, int(y * self.S)))
            clas = labels[b]

            #the amount at which the center is offset to the boxes
            dx = (x * self.S) - cell_x
            dy = (y * self.S) - cell_y

            #For multiple slots (B > 1), assign the first image to the first available slot
            for slot in range(self.B):
                if target[cell_y, cell_x, slot*5] == 0:
                    target[cell_y, cell_x, slot*5+0] = 1 #p_obj
                    target[cell_y, cell_x, slot*5+1] = dx #x_off
                    target[cell_y, cell_x, slot*5+2] = dy #y_off
                    target[cell_y, cell_x, slot*5+3] = w #width
                    target[cell_y, cell_x, slot*5+4] = h #height
                    break

            target[cell_y, cell_x, self.B*5 + clas] = 1
        
        img_tensor = self.transforms(img)

        return img_tensor, target



