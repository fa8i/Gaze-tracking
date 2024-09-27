
# Recopilación de Datos para Entrenamiento y Calibración en Seguimiento de la Mirada

Esta herramienta permite la recopilación de datos de mirada necesarios para la calibración personalizada o el entrenamiento de modelos de seguimiento ocular. Fue desarrollada como parte de la tesis de maestría de [P. Perle](https://github.com/pperle/gaze-tracking) sobre [seguimiento ocular con una webcam monocular](https://github.com/pperle/gaze-tracking). También está disponible el [framework completo del pipeline de seguimiento ocular](https://github.com/pperle/gaze-tracking-pipeline).

El resultado de esta herramienta es una carpeta que contiene un archivo CSV con la posición objetivo (en píxeles) donde la persona está mirando y el nombre del archivo de imagen correspondiente capturado con la webcam. Para obtener buenos resultados de calibración, se recomienda tomar al menos 9 imágenes de calibración, y mientras más, mejor.

## Instrucciones para ejecutar

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Si es necesario, calibra la cámara utilizando el script interactivo proporcionado:
   ```bash
   python calibrate_camera.py
   ```
   Puedes consultar más detalles sobre la calibración de la cámara en la documentación oficial de [OpenCV](https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html).

3. Para mayor precisión, también se recomienda calibrar la posición de la pantalla siguiendo el método descrito por [Takahashi et al.](https://doi.org/10.2197/ipsjtcva.8.11), quienes proveen una implementación en [OpenCV y Matlab](https://github.com/computer-vision/takahashi2012cvpr).

4. Ejecuta el script principal para iniciar la recopilación de datos:
   ```bash
   python main.py --base_path=./data/p00
   ```
   Este código ha sido probado en Ubuntu 20.10 y Ubuntu 21.04. Si usas macOS o Windows, puede que necesites proporcionar manualmente los parámetros del monitor, por ejemplo:
   ```bash
   --monitor_mm=750,420 --monitor_pixels=1920,1080
   ```
   Además, es posible que necesites ajustar los valores de `TargetOrientation` en el archivo `utils.py`.

5. Durante la recolección de datos, mira la pantalla y presiona la tecla de flecha correspondiente a la dirección en la que apunta la letra `E` cuando su color cambie de azul a naranja. Es recomendable presionar la tecla varias veces ya que OpenCV a veces no registra el primer clic.

6. Presiona la tecla `q` cuando la recopilación de datos haya finalizado.

7. Puedes visualizar los datos recopilados imagen por imagen ejecutando el siguiente comando:
   ```bash
   python visualization.py --base_path=./data
   ```

**Créditos**: Este código fue desarrollado originalmente por [P. Perle](https://github.com/pperle) como parte de su tesis de maestría. La herramienta ha sido adaptada para su uso en este proyecto con modificaciones para ajustarse a las necesidades específicas de integración en este proyecto.
Este código es complementario el proyecto, pero no necesario. Su utilización supone incrementar los datos de entrenamiento y vusualizarlos.