import sys
import os
import cv2

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))  # Adds root dir to path

from src.landmarker.mediapipe_landmarker import MediaPipeLandmarker
from src.landmarker.blendshape_logger import BlendshapeLogger, BlendshapePlotter
from src.utils.functions_utils import calculate_bounding_box, extract_angles

# Configuración del modelo y otros parámetros
model_path = 'src/models/face_landmarker.task'
csv_filename = 'blendshapes.csv'
blendshapes = False
transformation_matrixes = False
frame_interval = 30

face_margins = (-0.1, 0.0)
eye_margins = (0.25, 0.5)

def main():
    landmarker = MediaPipeLandmarker(
        detector_path=model_path, 
        tracker=True, 
        blendshapes=blendshapes, 
        transformation_matrixes=transformation_matrixes
    )
    cap = cv2.VideoCapture(0)

    if blendshapes:
        logger = BlendshapeLogger(filename=csv_filename, frame_interval=frame_interval)
        # plotter = BlendshapePlotter()
        print(f"BlendshapeLogger inicializado. Guardando cada {frame_interval} frames en {csv_filename}.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result, blends, matrix = landmarker.detect(frame)

        if result:
            face_bbox = calculate_bounding_box(result.all_landmarks, face_margins, aspect_ratio=1.0)

            right_eye_landmarks = (
                result.right_eye.upper_eyelid + result.right_eye.lower_eyelid +
                result.right_eye.inner_side + result.right_eye.outer_side)
            right_eye_bbox = calculate_bounding_box(right_eye_landmarks, eye_margins, aspect_ratio=2/1)

            left_eye_landmarks = (
                result.left_eye.upper_eyelid + result.left_eye.lower_eyelid +
                result.left_eye.inner_side + result.left_eye.outer_side)
            left_eye_bbox = calculate_bounding_box(left_eye_landmarks, eye_margins, aspect_ratio=2/1)

            # Dibujar los bounding boxes
            # Cara
            cv2.rectangle(
                frame, (face_bbox.x, face_bbox.y),
                (face_bbox.x + face_bbox.width, face_bbox.y + face_bbox.height),
                (255, 0, 0), 2)

            # Ojo derecho
            cv2.rectangle(
                frame, (right_eye_bbox.x, right_eye_bbox.y),
                (right_eye_bbox.x + right_eye_bbox.width, right_eye_bbox.y + right_eye_bbox.height),
                (0, 255, 0), 2)

            # Ojo izquierdo
            cv2.rectangle(
                frame,
                (left_eye_bbox.x, left_eye_bbox.y),
                (left_eye_bbox.x + left_eye_bbox.width, left_eye_bbox.y + left_eye_bbox.height),
                (0, 255, 0), 2)

        if blends and logger:
            logger.log(blends)
            # plotter.plot(blends)

        cv2.imshow('Face and Eyes Bounding Boxes', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
