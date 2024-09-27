from argparse import ArgumentParser
import cv2
import os
from glob import glob

def extract_images(base_path, output_folder):
    """Recorre las carpetas del dataset MPIIFaceGaze y extrae las caras de las imágenes, guardándolas en la carpeta de destino.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas 'pXX'.
        output_folder (str): Ruta donde se guardarán las imágenes recortadas.
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png')

    os.makedirs(output_folder, exist_ok=True)

    p_folders = [d for d in glob(os.path.join(base_path, 'p*')) if os.path.isdir(d)]
    for p_path in sorted(p_folders):
        day_folders = [d for d in glob(os.path.join(p_path, 'day*')) if os.path.isdir(d)]
        for day_path in sorted(day_folders):
            images = []
            for ext in image_extensions:
                images.extend(glob(os.path.join(day_path, ext)))
            for image_path in sorted(images):
                extract_face_crops(image_path, base_path, output_folder)

def extract_face_crops(image_path, base_path, output_folder):
    """Procesa una imagen individual para extraer la cara y guardarla en la carpeta de destino.

    Args:
        image_path (str): Ruta de la imagen a procesar.
        base_path (str): Ruta base para calcular rutas relativas.
        output_folder (str): Ruta donde se guardará la imagen recortada.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al leer la imagen: {image_path}")
        return

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar umbral para obtener máscara de píxeles no negros
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    # Encontrar los contornos de las áreas no negras
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = img[y:y+h, x:x+w]

        relative_path = os.path.relpath(os.path.dirname(image_path), base_path)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        image_file = os.path.basename(image_path)
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, cropped)
        print(f"Imagen guardada en: {output_path}")
    else:
        print(f"No se encontraron áreas no negras en la imagen: {image_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, help="Ruta donde se guarda las imágenes de MPIIFaceGaze")
    parser.add_argument("--output_path", "-o", type=str, help="Ruta donde se guardarán las imágenes")
    args = parser.parse_args()
    extract_images(args.input_path, args.output_path)
