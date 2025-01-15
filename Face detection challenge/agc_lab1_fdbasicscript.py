import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread
from scipy.io import loadmat
from typing import List, Union
import time
import dlib
import cv2


# ---------------------------------------------
# Function to Compute Detection Scores
# ---------------------------------------------
def compute_detection_scores(
    DetectionSTR: List[List[List[Union[int, float]]]],
    AGC_Challenge1_STR: pd.DataFrame,
    show_figures: bool = False,
) -> float:
    """
    Compute face detection scores (F1-score) for detected bounding boxes against ground truth.

    Parameters:
    DetectionSTR (List[List[List[Union[int, float]]]]):
        Detection results for each image. Each entry is a list of bounding boxes `[x1, y1, x2, y2]`.
    AGC_Challenge1_STR (pd.DataFrame):
        Ground truth DataFrame with 'imageName' (image paths) and 'faceBox' (bounding boxes).
    show_figures (bool):
        If True, displays images with ground truth and detected bounding boxes.

    Returns:
    float: Average F1-score across all images.
    """
    all_f1_scores = []

    for idx, row in AGC_Challenge1_STR.iterrows():
        gt_boxes = row["faceBox"]
        det_boxes = DetectionSTR[idx]

        # Handle edge cases where no boxes are present
        if len(gt_boxes) == 0 and len(det_boxes) == 0:
            all_f1_scores.append(1.0)  # Perfect score for no detections
            continue
        elif len(gt_boxes) == 0 or len(det_boxes) == 0:
            all_f1_scores.append(0.0)  # No match
            continue

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
        matched = iou_matrix > 0.5  # IoU threshold for a valid match
        true_positives = np.sum(matched.any(axis=1))
        false_positives = len(det_boxes) - true_positives
        false_negatives = len(gt_boxes) - true_positives

        # Compute precision, recall, and F1-score
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        all_f1_scores.append(f1_score)

        # Optionally visualize results
        if show_figures:
            image = imread(row["imageName"])
            fig, ax = plt.subplots()
            ax.imshow(image)
            for box in gt_boxes:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor="blue",
                        fill=False,
                    )
                )
            for box in det_boxes:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor="green",
                        fill=False,
                    )
                )
            plt.title(f"F1-score: {f1_score:.2f}")
            plt.show()

    # Return average F1-score across all images
    return np.mean(all_f1_scores)


# ---------------------------------------------
# Face Detection Function Placeholder
# ---------------------------------------------
def MyFaceDetectionFunction(A):
    """
    Student-implemented function for face detection.

    Parameters:
    A (np.ndarray): Input image.

    Returns:
    List[List[int]]: List of detected bounding boxes in the format [x1, y1, x2, y2]. [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    """

    #step2: converts to gray image
    gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

    #step3: get HOG face detector and faces
    hogFaceDetector = dlib.get_frontal_face_detector()
    faces = hogFaceDetector(gray, 1)

    bounding_boxes = []
    for rect in faces:
        x1 = rect.left()    # Coordenada x de la esquina superior izquierda
        y1 = rect.top()     # Coordenada y de la esquina superior izquierda
        x2 = rect.right()   # Coordenada x de la esquina inferior derecha
        y2 = rect.bottom()  # Coordenada y de la esquina inferior derecha
        
        # Añadir la bounding box a la lista
        bounding_boxes.append([x1, y1, x2, y2])

    
    return bounding_boxes


# ---------------------------------------------
# Basic Script for Face Detection Challenge
# ---------------------------------------------
# Load challenge training data
dir_challenge = "C:\\Users\\jgojg\\OneDrive\\Escritorio\\universidadportatil\\4to\\2ndotrim\\caretos\\Face-and-gesture-analysis\\Face detection challenge\\"
AGC_Challenge1_TRAINING = loadmat(dir_challenge + "fga_face_detection_training.mat")
AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING["AGC_Challenge1_TRAINING"])
AGC_Challenge1_TRAINING = [
    [row.flat[0] if row.size == 1 else row for row in line]
    for line in AGC_Challenge1_TRAINING
]
columns = ["id", "imageName", "faceBox"]
AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)

# Provide the path to the images
imgPath = "C:\\Users\\jgojg\\OneDrive\\Escritorio\\universidadportatil\\4to\\2ndotrim\\caretos\\l01_face_detection\\l01_face_detection\\training\\"
AGC_Challenge1_TRAINING["imageName"] = imgPath + AGC_Challenge1_TRAINING[
    "imageName"
].astype(str)

# Initialize detection results and timer
DetectionSTR = []
total_time = 0

# Process each image in the dataset
for idx, im in enumerate(AGC_Challenge1_TRAINING["imageName"]):
    A = imread(im)
    try:
        start_time = time.time()
        det_faces = MyFaceDetectionFunction(A)  # Call the student's function
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
    except Exception as e:
        print(f"Error processing image {im}: {e}")
        det_faces = []

    DetectionSTR.append(det_faces)

# Compute F1-score
FD_score = compute_detection_scores(
    DetectionSTR, AGC_Challenge1_TRAINING, show_figures=False
)

# Display final results
minutes, seconds = divmod(total_time, 60)
print(f"F1-score: {FD_score * 100:.2f}, Total time: {int(minutes)} m {seconds:.2f} s")
