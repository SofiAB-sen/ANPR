from ultralytics import YOLO
import torch
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import cv2
import glob 
from datetime import datetime
import argparse


import func

#Modelos
modelo_placas = YOLO("yolo11n_Plate_Recognition_v2.pt") 
modelo_caracteres = YOLO("yolo11n_OCR_v2.pt") 

parser = argparse.ArgumentParser(description='Detect plates and send results.')
parser.add_argument('--fecha_hora', required=True, help='Fecha y hora en formato YYYY-MM-DD HH:MM')
parser.add_argument('--ubicacion', required=True, help='Ubicación de la cámara')
parser.add_argument('--id_camara', required=True, help='ID de la cámara')
parser.add_argument('--image_path', required=True, help='Ruta a la imagen o carpeta de imágenes')

args = parser.parse_args()

#Obtener argumentos 
fecha_hora = args.fecha_hora
ubicacion = args.ubicacion
id_camara = args.id_camara
image_path = args.image_path

"""fecha_hora = input("Ingrese la fecha y hora (YYYY-MM-DD HH:MM): ") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ubicacion = input("Ingrese la ubicación: ")
id_camara = input("Ingrese el ID de la cámara: ")

image_path = input("Ingrese la ruta de la imagen o carpeta: ")"""

if not os.path.exists(image_path):
    print("La ruta no existe.")
    exit()
  
output_folder = "Placas_recortadas_yolo" #Carpeta de guardado placas recortadas
os.makedirs(output_folder, exist_ok=True)

tabla_resultados = "registro_placas.csv"
columnas = ["fecha_hora", "ubicacion", "camara_id", "ruta_placa", "placa"]
if os.path.exists(tabla_resultados):
    df = pd.read_csv(tabla_resultados)
else:
    pd.DataFrame(columns=columnas).to_csv(tabla_resultados, index=False)

def procesar_imagen(ruta_imagen):
    resultado = func.run_yolo_detector_placa(modelo_placas, modelo_caracteres, ruta_imagen, output_folder)
    if resultado:
        df = pd.read_csv(tabla_resultados)
        for ruta_placa, placa in zip(resultado["ruta_placa"], resultado["placa"]):
            nueva_fila = pd.DataFrame([[fecha_hora, ubicacion, id_camara, ruta_placa, placa]], columns=columnas)
            df = pd.concat([df, nueva_fila], ignore_index=True)
        df.to_csv(tabla_resultados, index=False)
        print(f"Resultados guardados en {tabla_resultados}")

#Si es carpeta, procesar todas las imágenes
if os.path.isdir(image_path):
    extensiones = [".jpg", ".jpeg", ".png"]
    imagenes = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_path) for f in filenames if os.path.splitext(f)[1].lower() in extensiones]
    for img in imagenes:
        procesar_imagen(img)
else:
    procesar_imagen(image_path)

print(f"✅ CSV actualizado: {tabla_resultados}")
print(f"✅ Placas recortadas guardadas en: {output_folder}")

