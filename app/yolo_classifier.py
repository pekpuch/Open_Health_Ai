import numpy as np
from ultralytics import YOLO
from PIL import Image


class YOLOClassifier:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def predict(self, image_path: str) -> int:
        image = Image.open(image_path)
        outputs = self.model(image, verbose=False)
        
        probs = outputs[0].probs
        print(probs)
        predicted_label = int(np.argmax(probs.data.cpu().detach().numpy()))
        
        return predicted_label
    
