import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image
from model import get_efficientnet  # Import EfficientNet model

# ✅ Define paths
MODEL_PATH = r"models/deeplock_EfficientNet_2.pth"

# ✅ Auto-detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Load trained EfficientNet model
print(f"📥 Loading model from: {MODEL_PATH}...")
model = get_efficientnet(pretrained=False).to(device)  # No need for pretrained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

# ✅ Define image preprocessing (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def predict_image(image_path):
    """Load an image, preprocess it, and predict if it's REAL or FAKE."""
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = transform(Image.fromarray(img)).unsqueeze(0).to(device)  # Preprocess

    # ✅ Run inference
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()  # Convert logits to probability
    
    prediction = "🛑 FAKE" if prob > 0.5 else "✅ REAL"
    print(f"🖼️ Image: {image_path} → Prediction: {prediction} (Confidence: {prob:.4f})")

# ✅ Example usage
if __name__ == "__main__":
    test_image = r"C:\Users\Aviral\Desktop\Minor Project\test images\fake\Screenshot 2025-02-17 213739.png"  # Change to the image path you want to test
    predict_image(test_image)