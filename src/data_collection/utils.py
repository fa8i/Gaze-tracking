import random
import sys
import os
import time
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
from typing import Tuple, Union

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_collection.webcam import WebcamSource


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Obtiene las dimensiones del monitor desde Gdk.

    Fuente: https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]] | Tuple[None, None]:
            Tupla con el ancho y alto del monitor en mm y píxeles, o (None, None) si no se pudo obtener.
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def create_image(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=1., orientation=TargetOrientation.RIGHT, target='E') -> Tuple[np.ndarray, float, bool]:
    """"
    Crea una imagen para mostrar en la pantalla.

    Args:
        monitor_pixels (Tuple[int, int]): Dimensiones del monitor en píxeles.
        center (Tuple[int, int], opcional): Centro del círculo y el texto. Por defecto es (0, 0).
        circle_scale (float, opcional): Escala del círculo. Por defecto es 1.0.
        orientation (TargetOrientation, opcional): Orientación del objetivo. Por defecto es TargetOrientation.RIGHT.
        target (str, opcional): Carácter a escribir en la imagen. Por defecto es 'E'.

    Returns:
        Tuple[np.ndarray, float, bool]:
            - Imagen creada como un arreglo de numpy.
            - Nueva escala reducida del círculo.
            - Booleano que indica si es el último frame en la animación.
    """
    width, height = monitor_pixels

    if orientation == TargetOrientation.LEFT or orientation == TargetOrientation.RIGHT:
        img = np.zeros((height, width, 3), np.float32)

        if orientation == TargetOrientation.LEFT:
            center = (width - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.LEFT:
            img = cv2.flip(img, 1)
    else:  # TargetOrientation.UP or TargetOrientation.DOWN
        img = np.zeros((width, height, 3), np.float32)
        center = (center[1], center[0])

        if orientation == TargetOrientation.UP:
            center = (height - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.UP:
            img = cv2.flip(img, 1)

        img = img.transpose((1, 0, 2))

    return img / 255, circle_scale * 0.9, end_animation_loop


def write_text_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Escribe el objetivo en la imagen y verifica si es el último frame de la animación.

    Args:
        center (Tuple[int, int]): Centro del círculo y el texto.
        circle_scale (float): Escala del círculo.
        img (np.ndarray): Imagen sobre la cual se escribirá.
        target (str): Carácter a escribir.

    Returns:
        bool: True si es el último frame de la animación y False en caso contrario.
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)
    cv2.circle(img, center, int(text_size[0] * 5 * circle_scale), (32, 32, 32), -1)
    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    end_animation_loop = circle_scale < random.uniform(0.1, 0.5)
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)

    return end_animation_loop


def get_random_position_on_screen(monitor_pixels: Tuple[int, int]) -> Tuple[int, int]:
    """
    Obtiene una posición válida aleatoria en el monitor.

    Args:
        monitor_pixels (Tuple[int, int]): Dimensiones del monitor en píxeles.

    Returns:
        Tuple[int, int]: Tupla con las coordenadas x e y aleatorias válidas en el monitor.
    """
    return int(random.uniform(0, 1) * monitor_pixels[0]), int(random.uniform(0, 1) * monitor_pixels[1])


def show_point_on_screen(window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:
    """
    Muestra un objetivo en la pantalla, ciclo completo de animación. Retorna datos recopilados si son válidos.

    Args:
        window_name (str): Nombre de la ventana donde se dibujará.
        base_path (str): Ruta donde se guardará la imagen.
        monitor_pixels (Tuple[int, int]): Dimensiones del monitor en píxeles.
        source (WebcamSource): Fuente de la webcam.

    Returns:
        Tuple[str, Tuple[int, int], float]:
            - Nombre del archivo de la imagen guardada.
            - Centro del objetivo en la pantalla.
            - Tiempo hasta la captura de la imagen.
    """
    circle_scale = 1.
    center = get_random_position_on_screen(monitor_pixels)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None

    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(10):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()

    if end_animation_loop:
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            if cv2.waitKey(42) & 0xFF == orientation.value:
                source.clear_frame_buffer()
                cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
                time_till_capture = time.time() - start_time_color_change
                break

    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(500)

    return f'{file_name}.jpg', center, time_till_capture
