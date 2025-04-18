import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from timm import create_model
import torch.nn.functional as F

# Set page configuration
st.set_page_config(
    page_title="DeepLock - Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# Define model paths
MODEL_DIR = "models/Final Models"  # Update this path to your model directory

# Model Architecture Definitions
class FrequencyAttention(nn.Module):
    """Frequency Attention Module for DFDetectV2"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DFDetectV2(nn.Module):
    """DFDetectV2 Model with ViT and Frequency Analysis"""
    def __init__(self):
        super().__init__()
        self.backbone = create_model('vit_base_patch16_224', pretrained=True)
        hidden_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove classification head

        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, hidden_dim, 1)
        )

        self.freq_attention = nn.ModuleList([FrequencyAttention(hidden_dim) for _ in range(3)])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Binary classification
        )

    def forward(self, x):
        vit_features = self.backbone(x)
        # Skip FFT for inference speed, use spatial features instead
        freq_features = self.freq_conv(x)
        B, C, H, W = freq_features.shape
        freq_features = freq_features.flatten(2).transpose(1, 2)

        features = vit_features + freq_features
        for attn in self.freq_attention:
            features = attn(features)

        cls_token = features[:, 0]
        output = self.head(cls_token)
        return output

def get_vit(pretrained=True):
    """Loads a pretrained ViT model and modifies it for binary classification."""
    model = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=1)
    if pretrained:
        num_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    return model

def get_efficientnet(pretrained=True):
    """Loads a pretrained EfficientNet-B3 model for binary classification."""
    model = create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=1,
        drop_path_rate=0.4,
        act_layer=nn.GELU
    )

    # Add classifier dropout
    if hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            model.classifier
        )
    elif hasattr(model, 'head'):
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            model.head
        )
    return model

def get_xception(pretrained=True):
    """Loads a pretrained Xception model for binary classification."""
    return create_model("xception", pretrained=pretrained, num_classes=1)

def get_dfdetect(pretrained=True):
    """Creates a DFDetectV2 model."""
    return DFDetectV2()

# Define model-specific transforms
def get_transform(model_name):
    """Get appropriate preprocessing transform for each model type"""
    base_size = 224
    
    if model_name == "EfficientNet":
        # EfficientNet uses ImageNet normalization
        return transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == "Xception":
        # Xception uses 299x299 images and different normalization
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # ViT and DFDetectV2 use standard normalization
        return transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

# Helper function to detect model type from state dict
def detect_model_type(state_dict):
    """Detect model architecture from state dict keys"""
    keys = list(state_dict.keys())
    
    # Check for EfficientNet specific keys
    if any(("efficientnet" in key) or ("blocks" in key and "bn" in key) for key in keys):
        return "EfficientNet", get_efficientnet(pretrained=False)
    
    # Check for Xception specific keys
    elif any("block" in key and "conv" in key and "bn" in key for key in keys):
        return "Xception", get_xception(pretrained=False)
    
    # Check for ViT specific keys
    elif any("blocks.0.norm1" in key or "patch_embed" in key for key in keys):
        if any("freq_attention" in key for key in keys):
            return "DFDetectV2", get_dfdetect(pretrained=False)
        else:
            return "ViT", get_vit(pretrained=False)
    
    # Default to ViT as fallback
    else:
        return "Unknown", get_vit(pretrained=False)

# Main app function
def main():
    st.title("🔍 DeepLock - Deepfake Image Detector")
    st.markdown("""
    Select a trained model and upload an image to check if it's **REAL or FAKE**.
    """)
    
    # Get list of available models
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
        model_files.insert(0, "Select Model")  # Add placeholder
    except FileNotFoundError:
        st.error(f"❌ Model directory '{MODEL_DIR}' not found. Please check the path.")
        model_files = ["Select Model"]
    
    # Model selection interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_model = st.selectbox("Choose a Model:", model_files, index=0)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Using device: {device}")
    
    # Initialize model variables
    model = None
    model_name = None
    model_scaling = 1.0  # Output scaling factor for certain models
    
    # Load selected model
    if selected_model != "Select Model":
        model_path = os.path.join(MODEL_DIR, selected_model)
        
        with st.spinner(f"Loading model: {selected_model}..."):
            try:
                # Load state dict to detect model type
                state_dict = torch.load(model_path, map_location=device)
                
                # Detect model type
                detected_model_name, model = detect_model_type(state_dict)
                model = model.to(device)
                
                # Apply scaling factor for Xception model (if needed)
                if "xception" in selected_model.lower() or detected_model_name == "Xception":
                    model_scaling = 0.1  # Scale Xception outputs
                
                # Try to load state dict with flexible loading
                try:
                    # First try direct loading
                    model.load_state_dict(state_dict)
                except Exception as e:
                    # If direct loading fails, try flexible loading
                    st.sidebar.warning(f"Standard loading failed, using flexible loading")
                    
                    # Filter out mismatched keys
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and model_dict[k].shape == v.shape}
                    
                    # Update model with matched keys
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict, strict=False)
                
                model.eval()
                model_name = detected_model_name
                st.sidebar.success(f"✅ Model loaded successfully!")
                st.sidebar.info(f"Detected model type: {model_name}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                model = None
                st.sidebar.error(f"Failed to load model: {str(e)}")
    
    # Image upload section
    with col2:
        uploaded_file = st.file_uploader("Upload an image to analyze:", type=["jpg", "jpeg", "png"])
    
    # Display interface based on model selection status
    if model is None:
        st.warning("⚠️ Please select a model before uploading an image.")
    
    # Process uploaded image
    if uploaded_file is not None and model is not None:
        # Create columns for image and results
        img_col, result_col = st.columns([1, 1])
        
        # Load and display image
        with img_col:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add analyze button
        analyze_button = st.button("🔍 Analyze Image")
        
        if analyze_button:
            try:
                # Get appropriate transform for the model
                transform = get_transform(model_name)
                
                # Preprocess image
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                    
                    # Apply model-specific scaling if needed
                    if model_scaling != 1.0:
                        output = output * model_scaling
                    
                    # Handle different output formats
                    if isinstance(output, tuple):
                        output = output[0]  # Some models return multiple outputs
                    
                    # Get probability (sigmoid for binary classification)
                    prob = torch.sigmoid(output).item()
                
                # Display results in the result column
                with result_col:
                    # Display result with color-coding
                    st.markdown("## Analysis Result")
                    
                    if prob > 0.3:
                        st.error(f"### 🚫 FAKE IMAGE DETECTED")
                        confidence = prob * 100
                    else:
                        st.success(f"### ✅ REAL IMAGE DETECTED")
                        confidence = (1 - prob) * 100
                    
                    # Create confidence gauge
                    st.markdown(f"### Confidence: {confidence:.1f}%")
                    
                    # Create a colorful progress bar
                    fake_percentage = prob * 100
                    
                    # Display progress bar
                    st.progress(prob)
                    
                    # Display fake/real percentages
                    st.markdown(f"""
                    - **Fake probability**: {fake_percentage:.1f}%
                    - **Real probability**: {100 - fake_percentage:.1f}%
                    """)
                    
                    # Display technical details in expander
                    with st.expander("Technical Details"):
                        st.write(f"Model type: {model_name}")
                        st.write(f"Raw model output: {output.cpu().numpy().flatten()[0]:.6f}")
                        if model_scaling != 1.0:
                            st.write(f"Output scaling factor: {model_scaling}")
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Make sure the image is in a standard format (JPG/PNG) and try again.")
    
    # Add information section
    st.markdown("---")
    st.markdown("""
    ### 📋 How to use:
    1. Select a trained model from the dropdown
    2. Upload an image you want to analyze
    3. Click "Analyze Image" to check if it's real or fake
    
    ### 🧠 Available Models:
    - **Xception**: Specialized CNN architecture with depth-wise separable convolutions
    - **EfficientNet-B3**: Efficient CNN with balanced accuracy and speed
    - **ViT**: Vision Transformer model that processes images as sequences of patches
    - **DFDetectV2**: Advanced model that combines transformer architecture with frequency analysis
    
    ### 🔍 About DeepLock:
    This application uses deep learning to detect manipulated or AI-generated images.
    """)

if __name__ == "__main__":
    main()