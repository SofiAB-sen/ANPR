import os
import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import numpy as np
import func  

#YOLO pretrained models
plates_model = YOLO("yolo11n_Plate_Recognition_v2.pt")
characters_model = YOLO("yolo11n_OCR_v2.pt")
vehicles_model = YOLO("yolo11n.pt")  


def detect_status_type_color(model, image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    clases = detections.boxes.cls.cpu().numpy()
    names = model.names

    status = "free"
    type = None
    color = None

    if len(boxes) > 0:
        
        box = boxes[0]
        class_id = int(clases[0])
        type = names[class_id]
        if type in ['car', 'truck', 'motorcycle', 'vehicle']:
            status = "ocuppied"

            x1, y1, x2, y2 = map(int, box)
            recorte = image_bgr[y1:y2, x1:x2]
            color = detect_color(recorte)
        else:
            type = 'Other'
            color = None

    return status, type, color


def detect_color(image):
    image = cv2.resize(image, (50, 50))
    data = image.reshape((-1, 3))
    data = np.float32(data)
    _, _, center = cv2.kmeans(data, 1, None, 
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                               10, cv2.KMEANS_RANDOM_CENTERS)
    b, g, r = center[0].astype(int)
    return color_to_name((r, g, b))


def color_to_name(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif abs(r - g) < 30 and b < 80:
        return "yellow"
    else:
        return "grey"


def analize_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "Invalid path"}

    status, type, color = detect_status_type_color(vehicles_model, image_path)
    plate_result = func.run_yolo_detector_plate(plates_model, characters_model, image_path, "plates_temp")
    plate = plate_result['plate'][0] if plate_result else None
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "plate": plate,
        "status": status,
        "date": date,
        "type": type,
        "color": color
        }

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Use: python Detect_complete.py path/to/image.jpg")
        exit()

    path_image = sys.argv[1]
    result = analize_image(path_image)
    print(json.dumps(result, indent=2, ensure_ascii=False))
