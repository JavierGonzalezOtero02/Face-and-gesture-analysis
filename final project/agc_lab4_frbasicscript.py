import os
import numpy as np
from imageio.v2 import imread
from scipy.io import loadmat
import random
import time
import itertools
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import dlib
from PIL import Image

##### INSTRUCTIONS TO RUN THIS FILE ####
# 1) Put yout dataset folder, its .mat and the model (.pth) in the same folder as this file
# 2) Fill the relative paths of below
# 3) Open a terminal in this folder
# 4) Run <python agc_lab4_frbasicscript.py>

mat_file_relative_path = r'\dataset_challenge_original\fga_challenge_train.mat' # Fill this variable with the relative path to your .mat file
dataset_relative_path = "\\dataset_challenge_original\\training\\" # Fill this variable with the relative path to your images dataset folder
model_rel_path = r'models\Student_shuffle.pth'


def compute_f1_score_for_recognition(pred_ids, true_ids, impostor_label=-1):
    """
    Computes the F1-score for face recognition as per the challenge rules.

    Args:
        pred_ids (list or array): Predicted identity values (in the original domain: -1 and 1..80).
        true_ids (list or array): Ground truth identity values (in the original domain).
        impostor_label (int, optional): The label representing impostors. Defaults to -1.

    Returns:
        float: The computed face recognition F1-score.
    """
    assert len(pred_ids) == len(true_ids), "Inputs must be of the same length"
    f_beta = 1

    # True Positives (TP): cases where the ground truth is not an impostor
    true_positive_indices = [
        i for i in range(len(true_ids)) if true_ids[i] != impostor_label
    ]
    nTP = sum(1 for i in true_positive_indices if pred_ids[i] == true_ids[i])

    # False Positives (FP): cases where a non-impostor identity is predicted but does not match the ground truth
    non_impostor_predictions = [
        i for i in range(len(pred_ids)) if pred_ids[i] != impostor_label
    ]
    nFP = sum(1 for i in non_impostor_predictions if pred_ids[i] != true_ids[i])

    # False Negatives (FN): cases where the ground truth is a real identity, but the model predicts impostor
    nFN = sum(
        1
        for i in range(len(true_ids))
        if pred_ids[i] == impostor_label and true_ids[i] != impostor_label
    )

    # F1-score calculation: 2*TP / (2*TP + FP + FN)
    denominator = 2 * nTP + nFP + nFN
    FR_score = (2 * nTP) / denominator if denominator > 0 else 0.0

    return FR_score

def MyFaceDetectionFunction(A):
    """
    Detects faces using dlib's HOG-based detector.
    """
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY) if len(A.shape) == 3 else A
    detections = detector(gray, upsample_num_times=1)
    return [[d.left(), d.top(), d.right(), d.bottom()] for d in detections]

def my_face_recognition_function(A, my_FRmodel):
    """
    Detects a face in image A, crops it, saves it temporarily, and classifies it using the model.
    
    Parameters:
    -----------
    A : np.ndarray
        Input image.
    my_FRmodel : torch.nn.Module
        Loaded face recognition model.
    
    Returns:
    --------
    int : ID of the recognized person (1-80) or -1 if not recognized.
    """
    
    # Detect faces in the image
    detected_faces = MyFaceDetectionFunction(A)

    # If no face is detected, return -1
    if len(detected_faces) == 0:
        return -1
    
    # Select the largest face (assuming it is the most relevant)
    detected_faces = sorted(detected_faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    x1, y1, x2, y2 = map(int, detected_faces[0])

    # Crop the detected face
    face_crop = A[y1:y2, x1:x2]
    if face_crop.size == 0:
        return -1  # If the crop is invalid

    # Load the image in PIL format for preprocessing
    image = Image.fromarray(face_crop).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Classification with the model
    with torch.no_grad():
        outputs = my_FRmodel(image)
        _, predicted = torch.max(outputs, 1)
    
    predicted_id = predicted.item()

    if predicted_id == 0:
        predicted_id = -1

    return predicted_id


# Basic script for Face Recognition Challenge
# --------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra

# Load challenge Training data
dir_challenge3 = os.getcwd()
AGC_Challenge3_TRAINING = loadmat(dir_challenge3 + mat_file_relative_path) # Modify 
AGC_Challenge3_TRAINING = np.squeeze(AGC_Challenge3_TRAINING["AGC_Challenge3_TRAINING"])

imageName = AGC_Challenge3_TRAINING["imageName"]
imageName = list(itertools.chain.from_iterable(imageName))

ids = list(AGC_Challenge3_TRAINING["id"])
ids = np.concatenate(ids).ravel().tolist()

faceBox = AGC_Challenge3_TRAINING["faceBox"]
faceBox = list(itertools.chain.from_iterable(faceBox))

imgPath = dir_challenge3 + dataset_relative_path

# Initialize results structure
AutoRecognSTR = []

# Initialize timer accumulator
total_time = 0

# Load your FRModel
class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        self.model = models.shufflenet_v2_x0_5(weights=False)  # Lightweight base model
        
        # Add additional layers to improve representation capacity
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to avoid overfitting
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

my_FRmodel_path = os.path.join(dir_challenge3, model_rel_path)
num_classes = 81
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize it
my_FRmodel = FaceClassifier(num_classes).to(device)  # Initialize the model
my_FRmodel.load_state_dict(torch.load(my_FRmodel_path, map_location=device))  # Load weights
my_FRmodel.eval()  # Set model to evaluation mode

# Image transformations
weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
default_transforms = weights.transforms()
'''
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
transform = default_transforms

for idx, im in enumerate(imageName):
    A = imread(imgPath + im)  # imgPath contains the full path to the images folder, im is A0001.jpg for example

    try:
        ti = time.time()
        # Timer on
        ###############################################################
        # Your face recognition function goes here. It must accept 2 input parameters:

        # 1. The input image A
        # 2. The recognition model

        # And must return a single integer number as output, which can be:

        # a) A number between 1 and 80 (representing one of the identities in the training set)
        # b) "-1" indicating that none of the 80 users is present in the input image

        autom_id = my_face_recognition_function(A, my_FRmodel)

        tt = time.time() - ti
        total_time = total_time + tt
    except Exception as e:
        # If the face recognition function fails, it will be assumed that no user was detected in this input image
        autom_id = random.randint(-1, 80)

    AutoRecognSTR.append(autom_id)
    # Print progress update every 100 images
    if (idx + 1) % 100 == 0 or (idx + 1) == len(imageName):
        print(f"Processed {idx + 1}/{len(imageName)} images")

FR_score = compute_f1_score_for_recognition(AutoRecognSTR, ids)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "F1-score: %.2f, Total time: %2d m %.2f s" % (100 * FR_score, int(minutes), seconds)
)
