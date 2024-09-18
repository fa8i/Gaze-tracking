import os
import time
import numpy as np
import cv2
import sys
import mediapipe as mp
from typing import Optional

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))  # Adds root dir to path

from src.utils.class_utils import LandmarkPoint, EyeLandmarks, LandmarkSet
from src.utils import constant_utils as cte

class MediaPipeLandmarker:
    """Esta clase implementa la detección de landmarks faciales con MediaPipe."""

    def __init__(self, detector_path: Optional[str] = None, tracker: bool = True, blendshapes: bool = False):
        """Inicializa el MediaPipeLandmarker.

        Args:
            blendshapes (bool, optional): Si es True, se habilitan los blendshapes.
            detector_path (str, optional): Ruta al modelo de MediaPipe. Si es None, se usa el modelo por defecto.
            tracker (bool, optional): Si es True, se utiliza el modo de seguimiento.
        """
        base_options = BaseOptions(model_asset_path=detector_path)
        running_mode = vision.RunningMode.VIDEO if tracker else vision.RunningMode.IMAGE
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.tracker = tracker
        self.detector = FaceLandmarker.create_from_options(options)
        self._first_timestamp = time.time()

    def detect(self, image: np.ndarray) -> Optional[LandmarkSet]:
        """Detecta los landmarks faciales en una imagen dada.

        Args:
            image (np.ndarray): Imagen donde buscar los landmarks. Debe ser una imagen en color (BGR).

        Returns:
            Tuple[Optional[LandmarkSet], Optional[FaceDetectionResult], Optional[dict]]:
                - LandmarkSet: Objeto con los landmarks detectados, o None si no se detecta ninguna cara.
                - FaceDetectionResult: Bounding box de la cara detectada, o None si no se detecta ninguna cara.
                - dict: Diccionario de blendshapes detectados, o None si no se detecta ninguna cara.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("La imagen proporcionada debe ser una imagen en color con 3 canales (BGR).")

        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self.tracker:
            timestamp = int(round((time.time() - self._first_timestamp) * 1000))
            detection_result = self.detector.detect_for_video(mp_image, timestamp)
        else:
            detection_result = self.detector.detect(mp_image)

        w = mp_image.width
        h = mp_image.height

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            all_landmarks = [LandmarkPoint(land.x * w, land.y * h, land.z * w) for land in landmarks]

            right_eye = EyeLandmarks(
                upper_eyelid=[all_landmarks[idx] for idx in cte.RIGHT_EYE_UPPER_LID],
                lower_eyelid=[all_landmarks[idx] for idx in cte.RIGHT_EYE_LOWER_LID],
                inner_side=[all_landmarks[idx] for idx in cte.RIGHT_EYE_INNER],
                outer_side=[all_landmarks[idx] for idx in cte.RIGHT_EYE_OUTER],
                eyebrow=[all_landmarks[idx] for idx in cte.RIGHT_EYE_BROW],
            )

            left_eye = EyeLandmarks(
                upper_eyelid=[all_landmarks[idx] for idx in cte.LEFT_EYE_UPPER_LID],
                lower_eyelid=[all_landmarks[idx] for idx in cte.LEFT_EYE_LOWER_LID],
                inner_side=[all_landmarks[idx] for idx in cte.LEFT_EYE_INNER],
                outer_side=[all_landmarks[idx] for idx in cte.LEFT_EYE_OUTER],
                eyebrow=[all_landmarks[idx] for idx in cte.LEFT_EYE_BROW],
            )

            blendshapes = {}
            if detection_result.face_blendshapes:
                face_blendshapes = detection_result.face_blendshapes[0]
                face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
                face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
                
                for key, value in zip(face_blendshapes_names, face_blendshapes_scores):
                    blendshapes[key] = value

            return LandmarkSet(all_landmarks, left_eye, right_eye), blendshapes
        return None, None