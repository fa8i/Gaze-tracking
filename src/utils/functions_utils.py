import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.class_utils import ROIBox, LandmarkPoint


def calculate_bounding_box(
    landmarks: List[LandmarkPoint],
    margins: Tuple[float, float] = (0.0, 0.0),
    aspect_ratio: Optional[float] = None,  # Relación de aspecto deseada (ancho / alto)
    image: Optional[np.ndarray] = None     # Imagen como array de NumPy para obtener dimensiones
) -> ROIBox:
    """Calcula el bounding box que engloba una lista de LandmarkPoint, ajustando con márgenes
    y opcionalmente ajustando la relación de aspecto deseada sin comprobar los límites de la imagen.

    Args:
        landmarks (List[LandmarkPoint]): Lista de puntos de landmarks.
        margins (Tuple[float, float]): Márgenes como porcentaje adicional para el ancho y alto.
        aspect_ratio (Optional[float]): Relación de aspecto deseada (ancho / alto). Si es None, no se ajusta.

    Returns:
        ROIBox: Bounding box ajustado.
    """
    if not landmarks:
        raise ValueError("La lista de landmarks está vacía.")

    # Convertir landmarks a array de NumPy para facilitar los cálculos
    points = np.array([[point.x, point.y] for point in landmarks])

    # Calcular los límites mínimos y máximos en x y y
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width, height = x_max - x_min, y_max - y_min

    # Aplicar márgenes
    width *= (1 + 2 * margins[0])
    height *= (1 + 2 * margins[1])

    # Calcular el centro del bbox original para recalcular x_min y y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    # Recalcular x_min y y_min basados en el nuevo ancho y alto con márgenes
    x_min, y_min = x_center - width / 2, y_center - height / 2

    # Ajustar la relación de aspecto si se proporciona
    if aspect_ratio is not None and height != 0:
        current_ratio = width / height
        if current_ratio < aspect_ratio:    # Aumentar ancho
            width = aspect_ratio * height
            x_min = x_center - width / 2
        elif current_ratio > aspect_ratio:  # Aumentar alto para cumplir la relación de aspecto
            height = width / aspect_ratio
            y_min = y_center - height / 2
        
    x_min=int(round(x_min))
    y_min=int(round(y_min))
    width=int(round(width))
    height=int(round(height))
    
    return ROIBox(x_min, y_min, width, height)


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """Graficar los blendshgapes en una gráfica de barras.

    Args:
        face_blendshapes (dict): Diccionario que contiene el nombre de los blendshapes y su valor normalizado.
    """
    face_blendshapes_names = face_blendshapes.keys()
    face_blendshapes_scores = face_blendshapes.values()
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def extract_angles(transformation_matrix):
    """Transforma una matriz de transformación 4x4 en ángulos de Euler (pitch, yaw, roll) y vector de traslación.
    
    Args:
        transformation_matrix (numpy.ndarray): Matriz de transformación 4x4.
        
    Returns:
        tuple:
            pitch (float): Ángulo de pitch.
            yaw (float): Ángulo de yaw.
            roll (float): Ángulo de roll.
            t (numpy.ndarray): Vector de traslación 3x1.
    """
    # Extract the rotation matrix (3x3) and the translation vector (3x1).
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # Calculate Euler angles
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0

    # pitch = np.degrees(pitch)
    # yaw = np.degrees(yaw)
    # roll = np.degrees(roll)
    
    return pitch, yaw, roll, t