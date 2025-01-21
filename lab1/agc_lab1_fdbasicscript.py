import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread
from scipy.io import loadmat
from typing import List, Union
import time
import cv2
import os
import dlib

# ---------------------------------------------
# Function to Compute Detection Scores
# ---------------------------------------------
from sklearn.metrics import confusion_matrix

def compute_detection_scores(
    DetectionSTR: List[List[List[Union[int, float]]]],
    AGC_Challenge1_STR: pd.DataFrame,
    show_figures: bool = False,
) -> float:
    all_f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Variables to store FP and FN images
    fp_images = []
    fn_images = []

    for idx, row in AGC_Challenge1_STR.iterrows():
        gt_boxes = row["faceBox"]
        det_boxes = DetectionSTR[idx]

        if len(gt_boxes) == 0 and len(det_boxes) == 0:
            all_f1_scores.append(1.0)
            continue
        elif len(gt_boxes) == 0 or len(det_boxes) == 0:
            all_f1_scores.append(0.0)
            continue

        # Convert bounding boxes to float64 for calculations
        gt_boxes = np.array(gt_boxes, dtype=np.float64)
        det_boxes = np.array(det_boxes, dtype=np.float64)

        # Calculate IoU for each ground-truth and detection pair
        iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)))

        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(det_boxes):
                # Compute intersection coordinates
                x1, y1 = max(gt[0], det[0]), max(gt[1], det[1])
                x2, y2 = min(gt[2], det[2]), min(gt[3], det[3])

                # Calculate areas
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                union = (
                    (gt[2] - gt[0]) * (gt[3] - gt[1])
                    + (det[2] - det[0]) * (det[3] - det[1])
                    - intersection
                )
                iou_matrix[i, j] = intersection / union if union > 0 else 0

        # Evaluate matches based on IoU threshold
        matched = iou_matrix > 0.5
        tp = np.sum(matched.any(axis=1))  # True Positives
        fp = len(det_boxes) - tp          # False Positives
        fn = len(gt_boxes) - tp           # False Negatives

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Store FP and FN images
        if fp > 0:  # False Positives
            fp_images.append(idx)
        if fn > 0:  # False Negatives
            fn_images.append(idx)

        # Compute precision, recall, and F1-score
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        all_f1_scores.append(f1_score)

        # Optionally visualize results for FP and FN
        if show_figures and (idx in fp_images or idx in fn_images):
            image = imread(row["imageName"])
            fig, ax = plt.subplots()
            ax.imshow(image)
            for box in gt_boxes:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor="blue",  # Ground truth in blue
                        fill=False,
                    )
                )
            for box in det_boxes:
                # Draw red rectangles for False Positives
                if idx in fp_images:
                    ax.add_patch(
                        Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            edgecolor="red",  # FP in red
                            fill=False,
                        )
                    )
                # Draw blue rectangles for False Negatives
                if idx in fn_images:
                    ax.add_patch(
                        Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            edgecolor="blue",  # FN in blue
                            fill=False,
                        )
                    )
            plt.title(f"F1-score: {f1_score:.2f}")
            plt.show()

    # Mostrar matriz de confusión total
    print("\nMatriz de confusión total:")
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")

    return np.mean(all_f1_scores)


# ---------------------------------------------
# Face Detection Function Placeholder
# ---------------------------------------------
def MyFaceDetectionFunction(A):
    """
    Detect faces using HOG-based dlib detector.

    Parameters:
    A (np.ndarray): Input image.

    Returns:
    List[List[int]]: List of detected bounding boxes in the format [x1, y1, x2, y2].
    """
    hog_detector = dlib.get_frontal_face_detector()

   
    # Verificar el número de canales de la imagen
    if len(A.shape) == 2:  # La imagen ya está en escala de grises
        gray = A
    elif len(A.shape) == 3:  # La imagen tiene 3 canales (RGB)
        gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Formato de imagen no reconocido: {A.shape}")

    # Evaluar el contraste de la imagen
    if np.std(gray) < 50:  # Umbral para determinar si el contraste es bajo
        # Aplicar CLAHE solo si el contraste es bajo
        clipLimit = 3.0  # Ajuste del clipLimit según sea necesario
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Detectar las caras usando HOG (Histogram of Oriented Gradients)
    detections = hog_detector(gray, upsample_num_times=1)

    face_boxes = [
        [d.left(), d.top(), d.right(), d.bottom()] for d in detections
    ]  # Convertir bounding boxes a lista

    return face_boxes

# ---------------------------------------------
# Basic Script for Face Detection Challenge
# ---------------------------------------------
# Directorio de trabajo
dir_challenge = os.path.dirname(os.getcwd())

# Cargar los datos de entrenamiento
AGC_Challenge1_TRAINING = loadmat(dir_challenge + "/fga_face_detection_training.mat")
AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING["AGC_Challenge1_TRAINING"])
AGC_Challenge1_TRAINING = [
    [row.flat[0] if row.size == 1 else row for row in line]
    for line in AGC_Challenge1_TRAINING
]
columns = ["id", "imageName", "faceBox"]
AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)
print(AGC_Challenge1_TRAINING.head())

# Ruta a las imágenes de entrenamiento
imgPath = os.path.join(dir_challenge, "training/")

# Actualizar las rutas de las imágenes
AGC_Challenge1_TRAINING["imageName"] = AGC_Challenge1_TRAINING["imageName"].apply(
    lambda x: os.path.join(imgPath, x)  # Asegura el formato correcto
)

# Verifica que los archivos existen
AGC_Challenge1_TRAINING = AGC_Challenge1_TRAINING[
    AGC_Challenge1_TRAINING["imageName"].apply(os.path.exists)
]

# Barajar el DataFrame antes de dividirlo en grupos
AGC_Challenge1_TRAINING = AGC_Challenge1_TRAINING.sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir el conjunto de entrenamiento en 6 grupos
groups = np.array_split(AGC_Challenge1_TRAINING, 6)

# Procesamiento y cálculo del F1-score para cada grupo
group_scores = []
for i, group in enumerate(groups):
    # Resetear índices del grupo para evitar problemas de indexación
    group = group.reset_index(drop=True)
    print(f"Procesando el grupo {i + 1}/{len(groups)} con {len(group)} imágenes...")

    DetectionSTR_group = []
    total_time_group = 0

    for idx, im in enumerate(group["imageName"]):
        A = imread(im)
        try:
            start_time = time.time()
            det_faces = MyFaceDetectionFunction(A)  # Llamar a la función de detección
            elapsed_time = time.time() - start_time
            total_time_group += elapsed_time
        except Exception as e:
            print(f"Error procesando imagen {im}: {e}")
            det_faces = []

        DetectionSTR_group.append(det_faces)

    # Calcular F1-score para el grupo
    FD_score_group = compute_detection_scores(DetectionSTR_group, group, show_figures=False)
    group_scores.append(FD_score_group)

    # Mostrar resultados del grupo
    minutes, seconds = divmod(total_time_group, 60)
    print(f"F1-score (grupo {i + 1}): {FD_score_group * 100:.2f}, Tiempo total: {int(minutes)} m {seconds:.2f} s")

# Mostrar los F1-scores finales de cada grupo
print("\nResultados finales por grupo:")
for i, score in enumerate(group_scores):
    print(f"Grupo {i + 1}: F1-score = {score * 100:.2f}%")



'''# Procesamiento de imágenes
DetectionSTR = []
total_time = 0

for idx, im in enumerate(AGC_Challenge1_TRAINING["imageName"]):
    if idx % 50 == 0 and idx > 0:  # Log progress every 50 images
        print(f"Processing image {idx}/{len(AGC_Challenge1_TRAINING)}...")

    A = imread(im)
    try:
        start_time = time.time()
        det_faces = MyFaceDetectionFunction(A)  # Llamar a la función de detección
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
    except Exception as e:
        print(f"Error procesando imagen {im}: {e}")
        det_faces = []

    DetectionSTR.append(det_faces)

# Calcular F1-score
FD_score = compute_detection_scores(DetectionSTR, AGC_Challenge1_TRAINING, show_figures=False)

# Mostrar resultados finales
minutes, seconds = divmod(total_time, 60)
print(f"F1-score: {FD_score * 100:.2f}, Tiempo total: {int(minutes)} m {seconds:.2f} s")'''