# ANPR

Este proyecto utiliza modelos YOLOv11 con Ultralytics para realizar reconocimiento automático de placas vehiculares (ANPR - Automatic Number Plate Recognition) a partir de imágenes. 

---

## ⚙️ Requisitos

### 🔹 Requisitos del sistema

- Python 3.10
- Conda o Miniconda

### 🔹 Crear entorno Conda

```bash
conda env create -f environment.yml
conda activate yolo-detect
```
Sin environment.yml:

```bash
conda create -n yolo-detect python=3.10
conda activate yolo-detect

# Dependencias básicas
conda install -c conda-forge numpy<2 pillow matplotlib tqdm pyyaml pandas seaborn pip requests opencv=4.8.0.76 

#Ultralytics y dependencias adicionales
pip install requests psutil py-cpuinfo "ultralytics-thop>=2.0.0"
```

🚀 Cómo usar

Para deteccion de tipo, placa, estado y color:

Ejecuta el script Detect_complete.py pasando la ruta de la imagen como argumento:

```bash
python3 Detect_complete.py /ruta/a/imagen.jpg
```

Para deteccion y recorte de placas:

Ejecuta el script Detect.py pasando los argumentos --date_time DATE_TIME --location LOCATION --id_camara ID_CAMARA --image_path IMAGE_PATH: 

```bash
python3 Detect.py --date_time 2025-08-09 --location Bello --id_camara Bello3 --image_path /Users/sarboledab/Downloads/carro4.jpg
```