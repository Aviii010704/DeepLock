import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Define paths
REAL_PATH = r"data/realtemptemp"
FAKE_PATH = r"data/faketemptemp"
SAVE_PATH = r"data/preprocessed_data_vit.pth"


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images
    transforms.RandomRotation(15),  # Rotate slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust lighting
    transforms.GaussianBlur(3),  # Add slight blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_and_preprocess_images(folder, label):
    images, labels = [], []
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = transform(img)  # Apply transformations
            images.append(img)
            labels.append(label)

    return torch.stack(images), torch.tensor(labels)

# Load and preprocess images
real_images, real_labels = load_and_preprocess_images(REAL_PATH, label=0)
fake_images, fake_labels = load_and_preprocess_images(FAKE_PATH, label=1)

# Combine real and fake data
X = torch.cat((real_images, fake_images), dim=0)
y = torch.cat((real_labels, fake_labels), dim=0)

# Save processed data
torch.save((X, y), SAVE_PATH)

print(f"✅ Preprocessing complete! Saved {X.shape[0]} images (Real: {real_labels.shape[0]}, Fake: {fake_labels.shape[0]})")
