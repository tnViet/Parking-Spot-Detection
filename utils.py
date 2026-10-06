import pickle
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize


EMPTY = True
NOT_EMPTY = False


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 16 * 16, out_features=2),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


def _load_model():
    pth_path = Path("model/best_cnn_model.pth")
    skl_path = Path("model/model.p")

    if pth_path.exists():
        model = ImageClassifier()
        state_dict = torch.load(pth_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model, "pytorch"
        print("PyTorch model loaded successfully")

    if skl_path.exists():
        with open(skl_path, "rb") as f:
            model = pickle.load(f)
        return model, "sklearn"
        print("Sklearn model loaded successfully")

    raise FileNotFoundError("Model not found. Expected 'model/best_cnn_model.pth' or 'model/model.p'.")


MODEL, MODEL_TYPE = _load_model()


def empty_or_not(spot_bgr):
    if MODEL_TYPE == "pytorch":
        print("Using PyTorch model")
        img_resized = resize(spot_bgr, (128, 128, 3), anti_aliasing=True)
        img_resized = img_resized.astype(np.float32)
        input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = MODEL(input_tensor)
            pred = int(torch.argmax(logits, dim=1).item())
        return EMPTY if pred == 0 else NOT_EMPTY

    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3), anti_aliasing=True)
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = int(MODEL.predict(flat_data)[0])
    return EMPTY if y_output == 0 else NOT_EMPTY


def get_parking_spots_from_json(json_path, video_w, video_h):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    orig_w = data.get('imageWidth')
    orig_h = data.get('imageHeight')
    
    scale_x = video_w / orig_w if orig_w else 1.0
    scale_y = video_h / orig_h if orig_h else 1.0
    
    spots = []
    for shape in data['shapes']:
        if shape['label'] != 'lot':
            continue
            
        pts = shape['points']
        shape_type = shape.get('shape_type', 'polygon')
        
        # Scale points
        scaled_pts = []
        for p in pts:
            scaled_pts.append([int(p[0] * scale_x), int(p[1] * scale_y)])
            
        if shape_type == 'rectangle':
            (x1, y1), (x2, y2) = scaled_pts
            scaled_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
        points_array = np.array(scaled_pts, dtype=np.int32)
        
        x, y, w, h = cv2.boundingRect(points_array)
        rect = cv2.minAreaRect(points_array)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        spots.append({
            'rotated_box': points_array,
            'rect': rect,
            'aabb': [x, y, w, h]
        })
    return spots


def get_parking_spots_bboxes(contours):
    spots = []
    for cnt in contours:

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        x, y, w, h = cv2.boundingRect(cnt)
        
        spots.append({
            'rotated_box': box,
            'rect': rect,
            'aabb': [x, y, w, h]
        })
    return spots