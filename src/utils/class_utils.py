from dataclasses import dataclass
from typing import List

@dataclass
class LandmarkPoint:
    """Clase que contiene las coordenadas x, y, z de cada landmark."""
    x: float
    y: float
    z: float

@dataclass
class EyeLandmarks:
    """Clase qie contiene listas de Landmarks de los ojos."""
    upper_eyelid: List[LandmarkPoint]
    lower_eyelid: List[LandmarkPoint]
    inner_side: List[LandmarkPoint]
    outer_side: List[LandmarkPoint]
    eyebrow: List[LandmarkPoint]

@dataclass
class LandmarkSet:
    """Clase que contiene el set de landmarks."""
    all_landmarks: List[LandmarkPoint]
    left_eye: EyeLandmarks
    right_eye: EyeLandmarks


@dataclass
class ROIBox:
    """Clase que contiene la Region de interes (Region Of Interest Box)."""
    x: float
    y: float
    width: float
    height: float


