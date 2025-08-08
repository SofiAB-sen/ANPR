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

#Models
plate_model = YOLO("yolo11n_Plate_Recognition_v2.pt") 
characters_model = YOLO("yolo11n_OCR_v2.pt") 

parser = argparse.ArgumentParser(description='Detect plates and send results.')
parser.add_argument('--date_time', required=True, help='Date and time in YYYY-MM-DD HH:MM format')
parser.add_argument('--location', required=True, help='Camera location')
parser.add_argument('--id_camara', required=True, help='Camera ID')
parser.add_argument('--image_path', required=True, help='Path to image file or folder')

args = parser.parse_args()

#Obtener argumentos 
date_time = args.date_time
location = args.location
id_camara = args.id_camara
image_path = args.image_path


if not os.path.exists(image_path):
    print("Teh path does not exists.")
    exit()
  
output_folder = "YOLO_plates" #Saving folder for cut plates
os.makedirs(output_folder, exist_ok=True)

results_table = "plate_register.csv"
columnas = ["date_time", "location", "camara_id", "plate_path", "plate"]
if os.path.exists(results_table):
    df = pd.read_csv(results_table)
else:
    pd.DataFrame(columns=columnas).to_csv(results_table, index=False)

def process_image(image_path):
    result = func.run_yolo_detector_plate(plate_model, characters_model, image_path, output_folder)
    if result:
        df = pd.read_csv(results_table)
        for plate_path, plate in zip(result["plate_path"], result["plate"]):
            new_row = pd.DataFrame([[date_time, location, id_camara, plate_path, plate]], columns=columnas)
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(results_table, index=False)
        print(f"results saved in {results_table}")

#if it is a folder, process every image
if os.path.isdir(image_path):
    extensions = [".jpg", ".jpeg", ".png"]
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_path) for f in filenames if os.path.splitext(f)[1].lower() in extensions]
    for img in images:
        process_image(img)
else:
    process_image(image_path)

print(f"✅ CSV updated: {results_table}")
print(f"✅ Plates saved in: {output_folder}")

