from ultralytics import YOLO
import torch
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import glob 



def run_yolo_detector_characteres(model, image_pil):
    results = model(image_pil)  
    detections = results[0].boxes
    top_boxes = sorted(detections, key=lambda box: float(box.conf), reverse=True)[:6]

    sorted_boxes = sorted(top_boxes, key=lambda box: float(box.xyxy[0][0]))

    plate = []
    for box in sorted_boxes:
        cls = int(box.cls)
        label = results[0].names[cls]
        plate.append(label)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"🔎 Detected '{label}' with {conf:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")

    plate = ''.join(plate)
    print(f"plate: {plate}")
    return plate

def run_yolo_detector_plate(model_plates, model_characteres, path, save_folder):
    
    image_pil = Image.open(path).convert("RGB")
    
    results = model_plates(path)  # Inference

    detections = results[0] 
    boxes = detections.boxes.xyxy.cpu().numpy()  
    scores = detections.boxes.conf.cpu().numpy()  
    class_ids = detections.boxes.cls.cpu().numpy()  
    names = model_plates.names  

    
    min_score = 0.4  # Minimum score threshold for detection
    filename = os.path.splitext(os.path.basename(path))[0]

    dic = {'plate_path': [], 'plate': []}
    for i, box in enumerate(boxes):
        score = scores[i]
        class_id = int(class_ids[i])
        class_name = names[class_id] 

        if score >= min_score and class_name.lower() in ["license plate", "vehicle registration plate", "plate"]:
            print(f"Plate found with {int(100 * score)}% confidence.")

            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = image_pil.crop((xmin, ymin, xmax, ymax))
            cropped_image = cropped_image.resize((96, 48))
            plate = run_yolo_detector_characteres(model_characteres, cropped_image)

            #SAVING DATA AND PLATE
            #save_path = os.path.join(save_folder, f"{filename}_plate_recortada_{plate}.jpg")
            #cropped_image.save(save_path)	
            #print(f"plate recortada y guardada: {save_path}")
            #dic['ruta_plate'].append(save_path)
            dic['plate'].append(plate)

    return dic if dic['plate'] else None
    

