import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime

#######################
#### MODEL TRAINER ####
#######################


#### INSTRUCTIONS TO RUN THIS FILE ####
# 1) Place cropped dataset folder INSIDE the same folder containing this file.
# 2) Place the .mat file corresponding to your cropped dataset folder inside the same folder containing this file
# 3) Fill the cropped_dataset_name variable with the NAME of your cropped dataset folder
# 3) Fill mat_file_name with the name of the .mat file corresponding to your cropped dataset folder
# 4) RUN python model.py
# 5) Model will be saved in the same folder of this file, with the name <face_classifier_{timestamp}.pth>
######################################################################################################################

cropped_dataset_name = r'cropped_dataset\training_cropped_full\training_cropped_full'
mat_file_name = r'cropped_dataset\fga_challenge_train_cropped_full.mat'
csv_file_name = r'cropped_dataset\train_cropped_no_teacher.csv'
import os
import pandas as pd

# 1️⃣ Load data from the .csv file
def load_data_from_csv(csv_file, image_folder):
    # Cargar el CSV con pandas
    df = pd.read_csv(csv_file)

    image_paths = []
    labels = []

    for _, row in df.iterrows():
        label = 0 if row["id"] == -1 else int(row["id"])  # Asignar etiqueta
        image_name = row["imageName"]  # Nombre de la imagen

        image_path = os.path.join(image_folder, image_name)  # Ruta completa
        if os.path.exists(image_path):  # Verificar si la imagen existe
            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels



# 1️⃣ Load data from the .mat file
def load_data_from_mat(mat_file, image_folder):
    mat_data = scipy.io.loadmat(mat_file)
    dataset_key = [key for key in mat_data.keys() if not key.startswith('__')][0]
    dataset = mat_data[dataset_key][0]  # Access data with the extra dimension

    image_paths = []
    labels = []

    for entry in dataset:
        if entry[0][0] == -1:
            label = 0                   # Person ID
        else: 
            label = int(entry[0][0])    # Person ID

        image_name = str(entry[1][0])   # Image name

        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):  # Ensure the image exists
            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels

# 2️⃣ Define custom Dataset
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 3️⃣ Preprocessing and DataLoader
weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
default_transforms = weights.transforms()
'''
# Enhanced transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Convert to Tensor before applying other transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # More variability
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),  # Add distortion
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Hide random parts
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Random blur
    default_transforms
])

# Transformations for validation
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    default_transforms
])
'''
dir_challenge3 = os.getcwd()
timestamp = datetime.now().strftime("%d_%H-%M-%S")

mat_file_path = dir_challenge3 + '/' + mat_file_name
csv_file_path = dir_challenge3 + '/' + csv_file_name
image_folder = os.path.join(dir_challenge3, cropped_dataset_name + '/')
model_save_path = os.path.join(dir_challenge3, f"face_classifier_{timestamp}.pth")

# image_paths, labels = load_data_from_mat(mat_file_path, image_folder)
image_paths, labels = load_data_from_csv(csv_file_path, image_folder)

# Split into training and validation
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

train_dataset = FaceDataset(train_paths, train_labels, default_transforms)# train_transform)
val_dataset = FaceDataset(val_paths, val_labels, default_transforms)# val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4️⃣ Define the CNN Model
class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        self.model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)  # Lightweight base model
        
        # Add additional layers to improve representational capacity
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes),
    
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),  # Dropout to prevent overfitting
                nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

num_classes = len(set(labels))  # Number of unique individuals
model = FaceClassifier(num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# **Print number of parameters**
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🔹 Total parameters in the model: {total_params}")
print(f"🔹 Trainable parameters: {trainable_params}")

# 5️⃣ Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    best_accuracy = 0  # To store the best model

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"🔹 Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        # Save the best model
        if save_path and accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model saved at {save_path}")

    print("✅ Training completed.")

# Run training with save option
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, save_path=model_save_path)
