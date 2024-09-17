
import csv
import os
from typing import Dict
import matplotlib.pyplot as plt

class BlendshapeLogger:
    """Clase para registrar blendshapes en un archivo CSV."""

    def __init__(self, filename: str, frame_interval: int = 1):
        """
        Inicializa el BlendshapeLogger.

        Args:
            filename (str): Nombre del archivo CSV donde se guardarán los datos.
            frame_interval (int, optional): Intervalo de frames para guardar los datos. Por defecto es 1 (guardar en cada frame).
        """
        self.filename = filename
        self.frame_interval = frame_interval
        self.current_frame = 0
        self.fieldnames = None

        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                pass 

    def log(self, blendshapes: Dict[str, float]):
        """
        Registra los blendshapes en el archivo CSV si corresponde según el intervalo de frames.

        Args:
            blendshapes (Dict[str, float]): Diccionario de blendshapes detectados.
        """
        self.current_frame += 1
        if self.current_frame % self.frame_interval != 0:
            return

        # Inicializar los encabezados en el primer registro
        if self.fieldnames is None:
            self.fieldnames = list(blendshapes.keys())
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

        # Escribir los datos
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(blendshapes)


class BlendshapePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 1.2)  # Fijar el límite del eje x entre 0 y 1.2
        self.ax.set_ylabel('Score', fontsize=14)
        self.ax.set_title('Blendshapes en Tiempo Real', fontsize=16)
        self.bars = None  # Almacenará las barras iniciales
        self.names = []
        self.fig.tight_layout()
        self.fig.show()

    def plot(self, blendshapes):
        names = list(blendshapes.keys())
        values = list(blendshapes.values())
        ranks = range(len(names))

        # Si es la primera vez que ploteamos, creamos las barras
        if self.bars is None:
            self.bars = self.ax.barh(ranks, values, color='blue')
            self.ax.set_yticks(ranks)
            self.ax.set_yticklabels(names, fontsize=12)
            self.ax.invert_yaxis()  # Para que los nombres vayan en el orden correcto
        else:
            # Actualizamos los valores de las barras ya existentes
            for bar, value in zip(self.bars, values):
                bar.set_width(value)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()