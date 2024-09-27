import argparse
import os
import cv2
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

# Agrega el directorio raíz al path
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.landmarker.mediapipe_landmarker import MediaPipeLandmarker
from src.utils.functions_utils import extract_angles

model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'models', 'face_landmarker.task'
))

if not os.path.exists(model_path):
    print(f"Modelo no encontrado en {model_path}")
    sys.exit(1)

landmarker = MediaPipeLandmarker(detector_path=model_path, tracker=False, transformation_matrixes=True)

def process_patient(patient_dir, output_dir):

    patient_id = os.path.basename(patient_dir)
    txt_file = os.path.join(patient_dir, f"{patient_id}.txt")
    if not os.path.exists(txt_file):
        print(f"Archivo no encontrado: {txt_file}")
        return
    
    # Lee los nombres de las imágenes y las coordenadas de la mirada desde el archivo de texto
    image_points = {}
    with open(txt_file, 'r') as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()
            if len(parts) >= 3:
                image_name = parts[0]
                x_pixel = float(parts[1])
                y_pixel = float(parts[2])
                image_points[image_name] = (x_pixel, y_pixel)
            else:
                print(f"La línea {line_num} en {txt_file} no tiene suficientes columnas.")
                continue
    
    calibration_dir = os.path.join(patient_dir, 'Calibration')
    screen_size_file = os.path.join(calibration_dir, 'screenSize.mat')
    if not os.path.exists(screen_size_file):
        print(f"Archivo no encontrado: {screen_size_file}")
        return

    screen_data = loadmat(screen_size_file)
    screen_data = {k: v for k, v in screen_data.items() if not k.startswith('_')}
    screen_df = pd.DataFrame({k: pd.Series(v.flatten()) for k, v in screen_data.items()})
    try:
        height_mm = float(screen_df['height_mm'].iloc[0])
        height_pixel = float(screen_df['height_pixel'].iloc[0])
        width_mm = float(screen_df['width_mm'].iloc[0])
        width_pixel = float(screen_df['width_pixel'].iloc[0])
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error al leer datos de tamaño de pantalla desde el archivo .mat: {e}")
        return

    data = []

    # Itera sobre cada directorio 'dayXX' en orden ascendente
    day_dirs = sorted([d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d)) and d.startswith('day')])
    for day_name in day_dirs:
        day_dir = os.path.join(patient_dir, day_name)
        image_files = sorted([f for f in os.listdir(day_dir) if os.path.isfile(os.path.join(day_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        for image_name in tqdm(image_files, desc=f"Procesando {patient_id}/{day_name}"):
            image_path = os.path.join(day_dir, image_name)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error al leer la imagen: {image_path}")
                continue

            relative_image_path = os.path.join(day_name, image_name)
            if relative_image_path not in image_points:
                print(f"Imagen {relative_image_path} no encontrada en el archivo {txt_file}.")
                continue

            x_pixel, y_pixel = image_points.get(relative_image_path, (None, None))
            x_mm = y_mm = None
            if x_pixel is not None and y_pixel is not None:
                x_mm = x_pixel * (width_mm / width_pixel)
                y_mm = y_pixel * (height_mm / height_pixel)

            # Aplica el landmarker para obtener la matriz de transformación
            _result, _blends, matrix = landmarker.detect(frame)
            pitch = yaw = roll = t0 = t1 = t2 = None
            if matrix is not None:
                pitch, yaw, roll, t = extract_angles(matrix)
                t0, t1, t2 = t.flatten()

            data.append({
                'image_name': relative_image_path,
                'X_pixel': x_pixel,
                'Y_pixel': y_pixel,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                't0': t0,
                't1': t1,
                't2': t2,
                'X(mm)': x_mm,
                'Y(mm)': y_mm,
                'height_mm': height_mm,
                'height_pixel': height_pixel,
                'width_mm': width_mm,
                'width_pixel': width_pixel
            })

    # Crea un DataFrame y lo guarda como un archivo CSV en la carpeta de datos
    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{patient_id}.csv")
    df.to_csv(csv_file, index=False)

def main(root_dir, output_dir):
    # Lista de directorios de pacientes
    patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('p')]
    
    for patient_dir in tqdm(patient_dirs, desc="Procesando pacientes"):
        process_patient(patient_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar imágenes de pacientes y extraer datos.")
    parser.add_argument(
        '-i',
        '--root_dir',
        type=str, 
        required=True,
        help='Directorio raíz que contiene las carpetas de pacientes (por ejemplo, p01, p02, ...)',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        required=False,
        default='.',
        help='Directorio para guardar los archivos CSV de salida. Por defecto es el directorio actual.',
    )
    args = parser.parse_args()
    main(args.root_dir, args.output_dir)
