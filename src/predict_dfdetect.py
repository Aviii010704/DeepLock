import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from model import get_dfdetect  # Import your trained model

# ✅ Configuration
MODEL_PATH = r"../models/deeplock_dfdetect_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load the trained model
print("📥 Loading model...")
model = get_dfdetect(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

# ✅ Define preprocessing (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Ensure correct input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_image(image_path):
    """Load an image, preprocess it, and predict if it's Real or Fake."""
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ Error: Image not found!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = transform(img).unsqueeze(0).to(DEVICE)  # Preprocess and add batch dim

    with torch.no_grad():
        output = model(img)  # Forward pass
        prob = torch.sigmoid(output).item()  # Convert logits to probability
    
    prediction = "Fake" if prob > 0.5 else "Real"
    print(f"🖼️ Image: {image_path} → Prediction: {prediction} (Confidence: {prob:.4f})")

# ✅ Example usage
if __name__ == "__main__":
    test_image = r"C:\Users\Aviral\Desktop\Minor Project_DFDetect\test images\fake\Screenshot 2025-02-18 031930.png"  # Change this to the image path
    predict_image(test_image)
