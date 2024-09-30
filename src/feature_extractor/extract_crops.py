import sys
import os
import cv2
from glob import glob
import argparse
from typing import List, Tuple

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))  # Adds root dir to path

from src.landmarker.mediapipe_landmarker import MediaPipeLandmarker
from src.utils.functions_utils import calculate_bounding_box

face_margins = (0.3, 0.15)
eye_margins = (0.2, 0.5)


def extract_face_crops(image_path: str, landmarker: MediaPipeLandmarker, output_path: str):
    """
    Extrae y guarda el recorte de la cara de una imagen.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        landmarker (MediaPipeLandmarker): Instancia de MediaPipeLandmarker para la detección de landmarks.
        output_path (str): Ruta donde se guardará el recorte de la cara.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    # Detectar landmarks
    result, _, _ = landmarker.detect(image)

    if not result:
        print(f"No se detectó una cara en la imagen: {image_path}")
        return

    # Calcular bounding box de la cara
    face_bbox = calculate_bounding_box(result.all_landmarks, margins=face_margins, aspect_ratio=1.0)

    # Recortar la cara
    x_min, y_min, width, height = face_bbox.x, face_bbox.y, face_bbox.width, face_bbox.height
    face_crop = image[y_min:y_min + height, x_min:x_min + width]

    # Guardar el recorte
    cv2.imwrite(output_path, face_crop)
    print(f"Guardado: {output_path}")

def extract_eyes_crops(image_path: str, landmarker: MediaPipeLandmarker, output_dir: str):
    """
    Extrae y guarda los recortes de los ojos de una imagen.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        landmarker (MediaPipeLandmarker): Instancia de MediaPipeLandmarker para la detección de landmarks.
        output_dir (str): Carpeta donde se guardarán los recortes de los ojos.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    result, _, _ = landmarker.detect(image)

    if not result:
        print(f"No se detectaron ojos en la imagen: {image_path}")
        return

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
    right_eye_bbox = calculate_bounding_box(right_eye_landmarks, margins=eye_margins, aspect_ratio=2/1)
    left_eye_bbox = calculate_bounding_box(left_eye_landmarks, margins=eye_margins, aspect_ratio=2/1)

    # Determinar cuál ojo es el derecho y cuál el izquierdo en la imagen
    if right_eye_bbox.x < left_eye_bbox.x:
        first_eye_bbox, first_eye_label = right_eye_bbox, "right_eye"
        second_eye_bbox, second_eye_label = left_eye_bbox, "left_eye"
    else:
        first_eye_bbox, first_eye_label = left_eye_bbox, "left_eye"
        second_eye_bbox, second_eye_label = right_eye_bbox, "right_eye"

    # Recortar y guardar cada ojo
    for bbox, label in [(first_eye_bbox, first_eye_label), (second_eye_bbox, second_eye_label)]:
        x_min, y_min, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        eye_crop = image[y_min:y_min + height, x_min:x_min + width]

        # Guardar el recorte
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_{label}{ext}")
        cv2.imwrite(output_path, eye_crop)
        print(f"Guardado: {output_path}")

def extract_images(base_path: str, output_folder: str, landmarker: MediaPipeLandmarker, features: List[str]):
    """
    Recorre las carpetas base_path para encontrar imágenes y extraer los recortes de la cara y/o los ojos.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas 'p*' y 'day*'.
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

                    # Definir las rutas de salida para los recortes
                    base_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(base_name)

                    if 'face' in features:
                        face_output_path = os.path.join(output_subdir, f"{name}_face{ext}")
                        extract_face_crops(image_path, landmarker, face_output_path)

                    if 'eyes' in features:
                        extract_eyes_crops(image_path, landmarker, output_subdir)

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
