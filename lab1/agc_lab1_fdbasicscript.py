'''

Lab01: Análisis de Gestos y Caras
Javier González Otero (243078)
Norbert Tomàs Escudero (242695)
Iria Quintero García (254373)

For executing, run the following command: python agc_lab1_fdbasicscript.py
The .py script, the .mat file and the directory with the images should all be on the same folder.

'''


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
def compute_detection_scores(
    DetectionSTR: List[List[List[Union[int, float]]]],
    AGC_Challenge1_STR: pd.DataFrame,
    show_figures: bool = False,
) -> float:
    """
    Computes the face detection F1-score for predicted bounding boxes versus ground truth.

    Parameters:
    -----------
    DetectionSTR : List[List[List[Union[int, float]]]]
        Detection results for each image. Each element is a list of bounding boxes,
        where each box is [x1, y1, x2, y2].
    AGC_Challenge1_STR : pd.DataFrame
        A DataFrame containing 'imageName' (paths to images) and 'faceBox' (ground truth bounding boxes).
    show_figures : bool
        If True, displays each image with ground truth boxes (blue) and detected boxes (green).

    Returns:
    --------
    float
        The mean F1-score across all images.
    """
    all_f1_scores = []

    for idx, row in AGC_Challenge1_STR.iterrows():
        # Convert ground truth and detections to int32 (or float) to avoid overflow
        gt_boxes = [np.array(box, dtype=np.int32) for box in row["faceBox"]]
        det_boxes = [np.array(box, dtype=np.int32) for box in DetectionSTR[idx]]

        # If there are no GT boxes and no detections, score is perfect (1.0 for that image)
        if len(gt_boxes) == 0 and len(det_boxes) == 0:
            all_f1_scores.append(1.0)
            continue
        # If one of them is empty but not the other, F1 = 0 for that image
        elif len(gt_boxes) == 0 or len(det_boxes) == 0:
            all_f1_scores.append(0.0)
            continue

        # Create a matrix for IoU (ground_truth_count x detection_count)
        iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)), dtype=np.float64)

        # Compute IoU for each (gt, det) pair
        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(det_boxes):
                # Basic validation of box dimensions
                if (
                    (gt[2] <= gt[0])
                    or (gt[3] <= gt[1])
                    or (det[2] <= det[0])
                    or (det[3] <= det[1])
                ):
                    print(f"Invalid box dimensions: GT {gt}, Det {det}")
                    continue

                # Intersection coordinates
                x1 = max(gt[0], det[0])
                y1 = max(gt[1], det[1])
                x2 = min(gt[2], det[2])
                y2 = min(gt[3], det[3])

                # Intersection width and height
                inter_w = max(0, x2 - x1)
                inter_h = max(0, y2 - y1)
                intersection = float(inter_w * inter_h)

                # Areas of GT and detection
                area_gt = float((gt[2] - gt[0]) * (gt[3] - gt[1]))
                area_det = float((det[2] - det[0]) * (det[3] - det[1]))
                union = area_gt + area_det - intersection

                iou = intersection / union if union > 0 else 0
                iou_matrix[i, j] = iou

        # --------------------------------------------------
        # GREEDY MATCHING STRATEGY
        # --------------------------------------------------
        matched_gts = set()
        true_positives = 0

        # For each detection, find the best matching GT (not matched yet)
        for det_idx in range(len(det_boxes)):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gts:
                    # This GT is already matched with another detection
                    continue

                current_iou = iou_matrix[gt_idx, det_idx]
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            # If the best IoU is above 0.5, we have a new true positive
            if best_iou > 0.5:
                true_positives += 1
                matched_gts.add(best_gt_idx)

        # The rest of detections that didn't match are false positives
        false_positives = len(det_boxes) - true_positives
        # The GT boxes that are not matched are false negatives
        false_negatives = len(gt_boxes) - true_positives

        # Compute precision, recall, and F1-score
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        all_f1_scores.append(f1_score)

        # Optionally show the image with boxes
        if show_figures:
            image = imread(row["imageName"])
            fig, ax = plt.subplots()
            ax.imshow(image)

            # Draw ground truth in blue
            for box in gt_boxes:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor="blue",
                        fill=False,
                        linewidth=2,
                    )
                )
            # Draw detections in green
            for box in det_boxes:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor="green",
                        fill=False,
                        linewidth=2,
                    )
                )

            plt.title(f"F1-score: {f1_score:.2f}")
            plt.show()

    # Compute average (mean) F1-score over all images
    mean_f1_score = np.mean(all_f1_scores) if len(all_f1_scores) > 0 else 0.0
    print(f"Mean F1-Score: {mean_f1_score:.4f}")

    return mean_f1_score


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

    # Check the number of image channels
    if len(A.shape) == 2:  # The image is already grayscale
        gray = A
    elif len(A.shape) == 3:  # The image has 3 channels (RGB)
        gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unrecognized image format: {A.shape}")

    # Evaluate image contrast
    if np.std(gray) < 50:  # Threshold to determine if contrast is low
        # Apply CLAHE only if the contrast is low
        clipLimit = 3.0  # Adjust clipLimit as necessary
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Detect faces using HOG (Histogram of Oriented Gradients)
    detections = hog_detector(gray, upsample_num_times=1)

    face_boxes = [
        [d.left(), d.top(), d.right(), d.bottom()] for d in detections
    ]  # Convert bounding boxes to list

    return face_boxes

# ---------------------------------------------
# Basic Script for Face Detection Challenge
# ---------------------------------------------
# Working directory
dir_challenge = os.getcwd()

# Load training data
AGC_Challenge1_TRAINING = loadmat(dir_challenge + "/AGC15_Challenge1_Test.mat")
AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING["AGC_Challenge1_TRAINING"])
AGC_Challenge1_TRAINING = [
    [row.flat[0] if row.size == 1 else row for row in line]
    for line in AGC_Challenge1_TRAINING
]
columns = ["id", "imageName", "faceBox"]
AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)

# Path to training images
imgPath = os.path.join(dir_challenge, "training/")

# Update image paths
AGC_Challenge1_TRAINING["imageName"] = AGC_Challenge1_TRAINING["imageName"].apply(
    lambda x: os.path.join(imgPath, x)  # Ensure correct format
)

# Verify that files exist
AGC_Challenge1_TRAINING = AGC_Challenge1_TRAINING[
    AGC_Challenge1_TRAINING["imageName"].apply(os.path.exists)
]

# Image processing
DetectionSTR = []
total_time = 0

for idx, im in enumerate(AGC_Challenge1_TRAINING["imageName"]):
    if idx % 50 == 0 and idx > 0:  # Log progress every 50 images
        print(f"Processing image {idx}/{len(AGC_Challenge1_TRAINING)}...")

    A = imread(im)
    try:
        start_time = time.time()
        det_faces = MyFaceDetectionFunction(A)  # Call detection function
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
    except Exception as e:
        print(f"Error processing image {im}: {e}")
        det_faces = []

    DetectionSTR.append(det_faces)

# Calculate F1-score
FD_score = compute_detection_scores(DetectionSTR, AGC_Challenge1_TRAINING, show_figures=False)

# Display final results
minutes, seconds = divmod(total_time, 60)
print(f"F1-score: {FD_score * 100:.2f}, Total time: {int(minutes)} m {seconds:.2f} s")
