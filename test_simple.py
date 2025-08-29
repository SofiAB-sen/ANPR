#!/usr/bin/env python3
"""
Versión con doble modelo YOLO:
- YOLOv8n -> detección general (vehículos, personas, etc.)
- keremberke/yolov8-license-plate-detection -> detección de matrículas
"""

import requests
import time
import datetime
import json
import re
import uuid
import os
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO
from PIL import Image
import cv2
#import pytesseract
from PIL import Image as PILImage

DAHUA_HOST = '192.168.1.108'
USERNAME = 'admin'
PASSWORD = 'ZonaZER2025*'

DELAY = 1
STILL_THRESHOLD = 3

# -------------------------------
# MODELOS
# -------------------------------
def load_models():
    try:
        vehicle_model = YOLO("yolo11n_Plate_Recognition_v2.pt")
        plate_model = YOLO("yolo11n_OCR_v2.pt")
        print("✅ Modelos YOLO cargados correctamente")
        return vehicle_model, plate_model
    except Exception as e:
        print(f"❌ Error al cargar modelos YOLO: {e}")
        return None, None

vehicle_model, plate_model = load_models()

# -------------------------------
# DETECCIÓN DE OBJETOS Y PLACAS
# -------------------------------
def detect_objects_in_image(image_path):
    """Detecta objetos en la imagen (vehículos, personas, etc.)"""
    if not vehicle_model:
        return [], []

    results = vehicle_model(image_path)
    detected_objects, vehicles_found = [], []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            bbox = box.xyxy[0].tolist()

            if confidence > 0.5:
                detected_objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": bbox
                })
                if class_name.lower() in ["car", "truck", "bus", "motorcycle"]:
                    vehicles_found.append({
                        "class": class_name,
                        "bbox": bbox,
                        "confidence": confidence
                    })
    return detected_objects, vehicles_found


def detect_plates_in_image(image_path):
    """Detecta placas en la imagen usando el modelo especializado"""
    if not plate_model:
        return []

    results = plate_model(image_path)
    plates = []

    image = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # OCR con modelo YOLO
        
                crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ocr_results = plate_model(crop_pil)
                ocr_boxes = ocr_results[0].boxes
                # Ordenar por la coordenada x para mantener el orden de lectura
                ocr_boxes_sorted = sorted(ocr_boxes, key=lambda box: float(box.xyxy[0][0]))
                chars = []
                for ocr_box in ocr_boxes_sorted:
                    cls = int(ocr_box.cls)
                    label = ocr_results[0].names[cls]
                    chars.append(label)
                text = ''.join(chars)

                plates.append({
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "text": text
                })

    return plates

# -------------------------------
# FUNCIONES PTZ
# -------------------------------
def get_ptz_position():
    try:
        url = f"http://{DAHUA_HOST}/cgi-bin/ptz.cgi?action=getStatus"
        response = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=5)
        return response.text if response.status_code == 200 else None
    except:
        return None


def parse_ptz_position(position_data):
    try:
        pan = float(re.search(r"status\.Postion\[0\]=([0-9.-]+)", position_data).group(1))
        tilt = float(re.search(r"status\.Postion\[1\]=([0-9.-]+)", position_data).group(1))
        zoom = float(re.search(r"status\.Postion\[2\]=([0-9.-]+)", position_data).group(1))
        preset_id = int(re.search(r"status\.PresetID=([0-9]+)", position_data).group(1))
        return {"pan": pan, "tilt": tilt, "zoom": zoom, "preset_id": preset_id}
    except:
        return {"pan": 0, "tilt": 0, "zoom": 0, "preset_id": 0}

# -------------------------------
# CAPTURA Y PROCESO
# -------------------------------
def capture_image_and_position():
    detection_uuid = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    url = f"http://{DAHUA_HOST}/cgi-bin/snapshot.cgi"
    response = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=10)

    if response.status_code != 200:
        print("❌ No se pudo capturar imagen")
        return

    image_filename = f"{detection_uuid}_PictureZoom.jpg"
    with open(image_filename, "wb") as f:
        f.write(response.content)

    print(f"📸 Imagen capturada: {image_filename}")

    # Detectar vehículos y placas
    detected_objects, vehicles_found = detect_objects_in_image(image_filename)
    detected_plates = detect_plates_in_image(image_filename)

    # Obtener posición PTZ
    position_data = get_ptz_position()
    parsed_position = parse_ptz_position(position_data) if position_data else {}

    # Construir JSON
    detection_data = {
        "uuid": detection_uuid,
        "timestamp": timestamp,
        "image_path": image_filename,
        "objects_detected": detected_objects,
        "vehicles_found": len(vehicles_found),
        "plates_detected": detected_plates,
        "plate_count": len(detected_plates),
        "coordinates": parsed_position
    }

    json_filename = f"{detection_uuid}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(detection_data, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON guardado: {json_filename}")
    print(f"🚗 Placas detectadas: {detected_plates}")


# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    last_position, still_count, is_stopped, last_capture_time = None, 0, False, 0
    print("🚀 Iniciando monitoreo PTZ con detección de objetos y placas...")

    while True:
        current_position = get_ptz_position()
        if current_position and current_position == last_position:
            still_count += 1
            if still_count == STILL_THRESHOLD and not is_stopped:
                print("🛑 PTZ SE DETUVO")
                capture_image_and_position()
                is_stopped, last_capture_time = True, time.time()
        else:
            if time.time() - last_capture_time > 5:
                still_count, last_position, is_stopped = 0, current_position, False
                print("🎥 PTZ en movimiento...")
        time.sleep(DELAY)


if __name__ == "__main__":
    main()
