# Proyecto de Fin de Máster: Gaze Tracking para Entrada de Escritura

Este proyecto forma parte del trabajo final de máster de **Data Science** realizado por **Fabián Melchor**, **Juan del Amo** y **Pablo Castillo**. El objetivo principal del proyecto es el análisis de diferentes métodos de *gaze tracking* (seguimiento de la mirada) y la implementación de un sistema de entrada de escritura mediante *computer vision* (visión por computadora) que permita a los usuarios escribir en un ordenador utilizando únicamente su mirada.

## Descripción del Proyecto

El proyecto se centra en el uso de la visión por computadora para desarrollar un sistema que permita a los usuarios interactuar con un teclado virtual a través del seguimiento de sus movimientos oculares. Esta tecnología, conocida como *gaze tracking*, es de gran utilidad en áreas como la accesibilidad, permitiendo que personas con movilidad reducida puedan escribir y controlar dispositivos con la mirada.

### Metas del Proyecto

1. **Análisis de Métodos de Gaze Tracking**: Evaluar los diferentes enfoques y algoritmos existentes para el seguimiento de la mirada, con el fin de seleccionar la metodología más adecuada para la implementación.
2. **Implementación de un Sistema de Entrada de Escritura**: Usar *gaze tracking* para permitir la entrada de texto en el ordenador, desarrollando un teclado virtual que el usuario pueda controlar con sus ojos.
3. **Computer Vision**: Utilizar técnicas avanzadas de visión por computadora para mejorar la precisión y usabilidad del sistema.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

src/ │ ├── demo/  │ └── main.py # Script principal para ejecutar el sistema de demostración. │ ├── landmarker/ │ ├── blendshape_logger.py # Registro y manejo de formas faciales para seguimiento ocular. │ └── mediapipe_landmarker.py # Implementación basada en la biblioteca MediaPipe para el tracking. │ ├── models/ │ ├── face_landmarker.task # Modelo de deteccion de landmarks de mediapipe. │ ├── utils/ │ ├── class_utils.py # Utilidades relacionadas con las clases y estructuras de datos. │ ├── constant_utils.py # Definición de constantes utilizadas en el proyecto. │ └── functions_utils.py # Funciones auxiliares utilizadas en diversos módulos. │ └── face_landmarker.task # Archivo de configuración para el reconocimiento facial.

### Dependencias

Para ejecutar el proyecto, asegúrate de instalar las dependencias necesarias. Puedes instalarlas ejecutando:

```bash
pip install -r requirements.txt
```

## Ejecución

El proyecto incluye un script principal que permite ejecutar la demostración del sistema de gaze tracking. Para correr la demo:

```bash
python src/demo/main.py
```

## Tecnologías Utilizadas

- Python: Lenguaje de programación principal.
- MediaPipe: Biblioteca para visión por computadora, utilizada para el seguimiento facial y ocular.
- OpenCV: Biblioteca de visión por computadora para procesamiento de imágenes y videos.
- Numpy: Biblioteca para operaciones matemáticas y manipulación de matrices.
- Pytorch: Biblioteca par elaborar y entrenar los modelos.

## Contribuidores
Fabián Melchor
Juan del Amo
Pablo Castillo
