from typing import List

class LandmarkPoint:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class EyeLandmarks:
    def __init__(
        self,
        upper_eyelid: List[LandmarkPoint],
        lower_eyelid: List[LandmarkPoint],
        inner_side: List[LandmarkPoint],
        outer_side: List[LandmarkPoint],
        eyebrow: List[LandmarkPoint],
    ):
        self.upper_eyelid = upper_eyelid
        self.lower_eyelid = lower_eyelid
        self.inner_side = inner_side
        self.outer_side = outer_side
        self.eyebrow = eyebrow

class LandmarkDetectionResult:
    def __init__(
        self,
        all_landmarks: List[LandmarkPoint],
        left_eye: EyeLandmarks,
        right_eye: EyeLandmarks,
    ):
        self.all_landmarks = all_landmarks
        self.left_eye = left_eye
        self.right_eye = right_eye

class FaceDetectionResult:
    def __init__(self, x, y, width, height):
        self.x = x      
        self.y = y      
        self.width = width
        self.height = height

