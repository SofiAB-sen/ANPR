import os
import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import numpy as np
import func  # Tu modulo anterior con run_yolo_detector_placa()

# Cargar modelos
modelo_placas = YOLO("yolo11n_Plate_Recognition_v2.pt")
modelo_caracteres = YOLO("yolo11n_OCR_v2.pt")
modelo_vehiculos = YOLO("yolo11n.pt")  # Este modelo debe detectar car, truck, motorcycle


def detectar_status_y_tipo_y_color(modelo, image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resultados = modelo(image_rgb)

    detections = resultados[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    clases = detections.boxes.cls.cpu().numpy()
    nombres = modelo.names

    status = "libre"
    tipo = None
    color = None

    if len(boxes) > 0:
        status = "ocupado"
        # Tomar el primer vehiculo detectado
        box = boxes[0]
        class_id = int(clases[0])
        tipo = nombres[class_id]

        x1, y1, x2, y2 = map(int, box)
        recorte = image_bgr[y1:y2, x1:x2]
        color = detectar_color_dominante(recorte)

    return status, tipo, color


def detectar_color_dominante(imagen):
    imagen = cv2.resize(imagen, (50, 50))
    data = imagen.reshape((-1, 3))
    data = np.float32(data)
    _, _, centro = cv2.kmeans(data, 1, None, 
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                               10, cv2.KMEANS_RANDOM_CENTERS)
    b, g, r = centro[0].astype(int)
    return convertir_color_a_nombre((r, g, b))


def convertir_color_a_nombre(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "blanco"
    elif r < 50 and g < 50 and b < 50:
        return "negro"
    elif r > g and r > b:
        return "rojo"
    elif g > r and g > b:
        return "verde"
    elif b > r and b > g:
        return "azul"
    elif abs(r - g) < 30 and b < 80:
        return "amarillo"
    else:
        return "gris"


def analizar_imagen(image_path):
    if not os.path.exists(image_path):
        return {"error": "Ruta no válida"}

    status, tipo, color = detectar_status_y_tipo_y_color(modelo_vehiculos, image_path)
    resultado_placa = func.run_yolo_detector_placa(modelo_placas, modelo_caracteres, image_path, "placas_temp")
    placa = resultado_placa['placa'][0] if resultado_placa else None
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resultado = {
        "plate": placa,
        "status": status,
        "fecha": fecha,
        "extras": {
            "color": color,
            "marca": None,
            "type": tipo
        }
    }
    return resultado


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python analyze_image.py ruta/a/imagen.jpg")
        exit()

    ruta_imagen = sys.argv[1]
    resultado = analizar_imagen(ruta_imagen)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
