import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import get_xception, get_vit, get_dfdetect, get_efficientnet, DFDetectV2

# Define path to model weights
MODEL_DIR = "models/Final Models"

# Load list of .pth files
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
model_files.insert(0, "Select Model")

# Streamlit UI
st.title("🔍 DeepLock - Deepfake Image Detector")
st.write("Select a trained model and upload an image to check if it's **REAL or FAKE**.")

# Dropdown to select model
selected_model = st.selectbox("Choose a Model:", model_files)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loader
model = None
model_type = None

if selected_model != "Select Model":
    model_path = os.path.join(MODEL_DIR, selected_model)

    # Identify model type by filename
    if "xception" in selected_model.lower():
        model = get_xception(pretrained=False).to(device)
        model_type = "Xception"
    elif "vit" in selected_model.lower():
        model = get_vit(pretrained=False).to(device)
        model_type = "ViT"
    elif "efficientnet" in selected_model.lower():
        model = get_efficientnet(pretrained=False).to(device)
        model_type = "EfficientNet"
    elif "dfdetect" in selected_model.lower():
        model = get_dfdetect(pretrained=False).to(device)
        model_type = "DFDetectV2"
    else:
        st.error("❌ Unsupported model selected.")

    # Load weights
    with st.spinner(f"📥 Loading model `{selected_model}`..."):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    st.success(f"✅ Model `{selected_model}` ({model_type}) loaded successfully!")

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Upload image and predict
if model:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Identify"):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()

            prediction = "🛑 FAKE" if prob > 0.5 else "✅ REAL"
            st.subheader(f"Prediction: {prediction}")
            st.write(f"**Confidence Score:** `{prob:.4f}`")
else:
    st.warning("⚠ Please select a model before uploading an image.")
