
# Procesamiento de Datos del Dataset MPIIFaceGaze

Este script sirve para procesar los datos obtenidos del dataset MPIIFaceGaze, que se puede descargar desde el siguiente enlace: [MPIIFaceGaze Dataset](https://perceptualui.org/research/datasets/MPIIFaceGaze/).

## Descripción del Script

El script procesa imágenes de pacientes y extrae datos relevantes para el seguimiento ocular, utilizando el landmarker de MediaPipe. Los datos extraídos incluyen información sobre la posición de los ojos en píxeles y milímetros, así como los ángulos de rotación de la cabeza (pitch, yaw, roll) y vector de traslación. El procesamiento se realiza de la siguiente manera:

1. **Lectura de datos**: Se leen las imágenes del paciente y las coordenadas de la mirada desde archivos de texto (`.txt`).
2. **Calibración de pantalla**: El script carga los datos de tamaño de pantalla desde archivos `.mat` para convertir las coordenadas de píxeles a milímetros.
3. **Aplicación de Landmarker**: El modelo MediaPipe se utiliza para detectar la cara y extraer los ángulos de rotación de la cabeza y la matriz de transformación.
4. **Generación de CSV**: Se crea un archivo CSV con todos los datos extraídos para cada paciente.

## Instrucciones de Uso

Para ejecutar el script, debes seguir los siguientes pasos:

1. Clona o descarga el código y asegúrate de que tienes instaladas las dependencias necesarias (ver `requirements.txt`).
2. Asegúrate de tener el dataset MPIIFaceGaze descargado y organizado adecuadamente.
3. Ejecuta el script con los siguientes argumentos:

```bash
python extract_metadata.py --root_dir /ruta/a/tu/directorio_de_pacientes --output_dir /ruta/de/salida
```

### Argumentos

- `--root_dir`: Directorio raíz que contiene las carpetas de los pacientes (por ejemplo, `p01`, `p02`).
- `--output_dir`: Directorio donde se guardarán los archivos CSV de salida. Si no se especifica, los archivos se guardarán en el directorio actual.

