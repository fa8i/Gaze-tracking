import sys
import os
import cv2
import numpy as np
from glob import glob
import argparse
from typing import List

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))) 

from src.landmarker.mediapipe_landmarker import MediaPipeLandmarker
from src.utils.functions_utils import calculate_bounding_box

face_margins = (-0.1, 0.0)
eye_margins = (0.25, 0.5)


def extract_crops(image_path: str, landmarker: MediaPipeLandmarker, output_dir: str, features: List[str]):
    """
    Extrae y guarda los recortes de la cara y los ojos de una imagen, rellenando con negro si los bounding boxes salen de los límites.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        landmarker (MediaPipeLandmarker): Instancia de MediaPipeLandmarker para la detección de landmarks.
        output_dir (str): Carpeta donde se guardarán los recortes de la cara y los ojos.
        features (List[str]): Lista de características a extraer ('face', 'eyes').
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    # Detectar landmarks de una vez
    result, _, _ = landmarker.detect(image)

    if not result:
        print(f"No se detectó una cara en la imagen: {image_path}")
        return

    # Obtener la ruta relativa y crear directorios de salida
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_subdir = os.path.join(output_dir)
    os.makedirs(output_subdir, exist_ok=True)

    # Extraer y guardar el recorte de la cara si está en las características
    if 'face' in features:
        extract_and_save_face_crop(image, result.all_landmarks, output_subdir, name, ext)

    # Extraer y guardar los recortes de los ojos si está en las características
    if 'eyes' in features:
        extract_and_save_eyes_crops(image, result, output_subdir, name, ext)


def extract_and_save_face_crop(image: np.ndarray, landmarks, output_subdir: str, name: str, ext: str):
    """
    Extrae el recorte de la cara y lo guarda, rellenando con negro si el bounding box sale de los límites.
    """
    # Calcular bounding box de la cara
    face_bbox = calculate_bounding_box(landmarks, margins=face_margins, aspect_ratio=1.0)

    # Recortar y ajustar la cara, rellenando con negro si es necesario
    face_crop = crop_and_pad(image, face_bbox)

    # Verificar si el recorte está vacío
    if np.count_nonzero(face_crop) == 0:
        print(f"El recorte de la cara está vacío: {name}{ext}")
        return

    face_output_path = os.path.join(output_subdir, f"{name}-full_face{ext}")
    cv2.imwrite(face_output_path, face_crop)
    print(f"Guardado: {face_output_path}")


def extract_and_save_eyes_crops(image: np.ndarray, result, output_subdir: str, name: str, ext: str):
    """
    Extrae los recortes de los ojos y los guarda, rellenando con negro si los bounding boxes salen de los límites.
    """
    # Obtener landmarks para cada ojo
    right_eye_landmarks = (
        result.right_eye.upper_eyelid + result.right_eye.lower_eyelid +
        result.right_eye.inner_side + result.right_eye.outer_side
    )
    left_eye_landmarks = (
        result.left_eye.upper_eyelid + result.left_eye.lower_eyelid +
        result.left_eye.inner_side + result.left_eye.outer_side
    )

    # Calcular bounding boxes para cada ojo
    right_eye_bbox = calculate_bounding_box(right_eye_landmarks, margins=eye_margins, aspect_ratio=3/2)
    left_eye_bbox = calculate_bounding_box(left_eye_landmarks, margins=eye_margins, aspect_ratio=3/2)

    # Determinar cuál ojo es el derecho y cuál el izquierdo en la imagen
    if right_eye_bbox.x < left_eye_bbox.x:
        eye_boxes = [(right_eye_bbox, "right_eye"), (left_eye_bbox, "left_eye")]
    else:
        eye_boxes = [(left_eye_bbox, "left_eye"), (right_eye_bbox, "right_eye")]

    # Recortar y guardar cada ojo
    for bbox, label in eye_boxes:
        eye_crop = crop_and_pad(image, bbox)

        # Verificar si el recorte está vacío
        if np.count_nonzero(eye_crop) == 0:
            print(f"El recorte del {label} está vacío: {name}{ext}")
            return

        # Guardar el recorte
        eye_output_path = os.path.join(output_subdir, f"{name}-{label}{ext}")
        cv2.imwrite(eye_output_path, eye_crop)
        print(f"Guardado: {eye_output_path}")


def crop_and_pad(image: np.ndarray, bbox) -> np.ndarray:
    """
    Recorta la región de una imagen dada por el bounding box. Si la región sale de los límites de la imagen,
    rellena con negro para mantener el tamaño original del bounding box.

    Args:
        image (np.ndarray): Imagen original.
        bbox: Bounding box que define la región a recortar.

    Returns:
        np.ndarray: Imagen recortada con el tamaño original del bounding box y relleno negro si es necesario.
    """
    h, w, _ = image.shape

    # Ajustar coordenadas si están fuera de los límites de la imagen
    x_min_corrected = max(bbox.x, 0)
    y_min_corrected = max(bbox.y, 0)
    x_max_corrected = min(bbox.x + bbox.width, w)
    y_max_corrected = min(bbox.y + bbox.height, h)

    # Recortar la imagen dentro de los límites
    crop = image[y_min_corrected:y_max_corrected, x_min_corrected:x_max_corrected]

    # Crear una imagen vacía (negra) del tamaño original del bounding box
    full_crop = np.zeros((bbox.height, bbox.width, 3), dtype=np.uint8)

    # Pegar el recorte dentro de la imagen negra en la posición correcta
    insert_x = x_min_corrected - bbox.x  # Si se salió por la izquierda, el valor será positivo
    insert_y = y_min_corrected - bbox.y  # Si se salió por arriba, el valor será positivo
    full_crop[insert_y:insert_y + (y_max_corrected - y_min_corrected),
              insert_x:insert_x + (x_max_corrected - x_min_corrected)] = crop

    return full_crop


def extract_images(base_path: str, output_folder: str, landmarker: MediaPipeLandmarker, features: List[str]):
    """
    Recorre las carpetas en base_path para encontrar imágenes y extraer los recortes de la cara y/o los ojos.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de imágenes.
        output_folder (str): Carpeta donde se guardarán los recortes.
        landmarker (MediaPipeLandmarker): Instancia de MediaPipeLandmarker para la detección de landmarks.
        features (List[str]): Lista de características a extraer ('face', 'eyes').
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png')

    p_folders = [d for d in glob(os.path.join(base_path, 'p*')) if os.path.isdir(d)]
    for p_path in sorted(p_folders):
        day_folders = [d for d in glob(os.path.join(p_path, 'day*')) if os.path.isdir(d)]
        for day_path in sorted(day_folders):
            for ext in image_extensions:
                for image_path in sorted(glob(os.path.join(day_path, ext))):
                    # Obtener la ruta relativa de la imagen respecto al base_path
                    relative_path = os.path.relpath(image_path, base_path)
                    relative_dir = os.path.dirname(relative_path)

                    # Crear el mismo directorio en el output_folder
                    output_subdir = os.path.join(output_folder, relative_dir)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Extraer recortes de la cara y/o los ojos según las características solicitadas
                    extract_crops(image_path, landmarker, output_subdir, features)

def main():
    parser = argparse.ArgumentParser(description="Extraer recortes de la cara y/o los ojos de imágenes en carpetas.")
    parser.add_argument("--input_dir", "-i", type=str, help="Directorio de entrada donde se encuentran las imágenes.")
    parser.add_argument("--output_dir", "-o", type=str, help="Directorio de salida donde se guardarán los recortes.")
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        nargs='+',
        choices=['eyes', 'face'],
        default=['eyes'],
        help="Características a extraer: 'eyes', 'face' o ambas. Escribir sin comas."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    features = args.features

    # Inicializar el landmarker
    model_path = '../models/face_landmarker.task'
    landmarker = MediaPipeLandmarker(
        detector_path=model_path,
        tracker=False,
        blendshapes=False,
        transformation_matrixes=False
    )

    extract_images(input_dir, output_dir, landmarker, features)


if __name__ == "__main__":
    main()
