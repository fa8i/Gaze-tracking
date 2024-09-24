import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.class_utils import ROIBox, LandmarkPoint


def calculate_bounding_box(
    landmarks: List[LandmarkPoint],
    vertical_margin: float = 0.0,
    horizontal_margin: float = 0.0
) -> ROIBox:
    """
    Calcula el bounding box dado una lista de LandmarkPoint, ajustando el tamaño según los márgenes proporcionados.

    Args:
        landmarks (List[LandmarkPoint]): Lista de puntos de landmarks.
        vertical_margin (float): Porcentaje a añadir por arriba y por abajo del bounding box original.
        horizontal_margin (float): Porcentaje a añadir a la izquierda y derecha del bounding box original.

    Returns:
        ROIBox: Bounding box que engloba todos los puntos proporcionados, ajustado con los márgenes.
    """
    points = np.array([[point.x, point.y] for point in landmarks])
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    width = x_max - x_min
    height = y_max - y_min

    # Calcular el centro del bounding box original
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Ajustar el ancho y alto según los márgenes
    width *= (1 + 2 * horizontal_margin)
    height *= (1 + 2 * vertical_margin)

    # Recalcular x_min y y_min basados en el nuevo ancho y alto
    x_min = x_center - width / 2
    y_min = y_center - height / 2

    # Asegurar que las coordenadas no salen de los límites de la imagen (opcional)
    # x_min = max(0, x_min)
    # y_min = max(0, y_min)

    # Convertir a enteros
    x_min = int(round(x_min))
    y_min = int(round(y_min))
    width = int(round(width))
    height = int(round(height))

    return ROIBox(x_min, y_min, width, height)


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """
    Graficar los blendshgapes en una gráfica de barras.

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
    """
    Transforma una matriz de transformación 4x4 en ángulos de Euler (pitch, yaw, roll) y vector de traslación.
    
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