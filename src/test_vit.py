import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from colorama import Fore, Style
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from einops import rearrange
from einops.layers.torch import Rearrange

class TestConfig:
    # Data paths
    DATA_PATH = "data/preprocessed_data_p1.pth"  # Main data path
    TEST_DATA_PATH = "data/preprocessed_data_test.pth"  # Optional separate test data
    
    # Model paths
    MODEL_DIR = "models/Final Models"
    MODEL_PATH = os.path.join(MODEL_DIR, "deeplock_vit.pth")
    
    # Output directories
    RESULTS_DIR = "test_results/vit"
    METRICS_PATH = os.path.join(RESULTS_DIR, "test_metrics.json")
    
    # Testing parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Add learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Add learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = nn.functional.softmax(att, dim=-1)
        att = self.att_drop(att)
        
        x = (att @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """Vision Transformer for binary classification."""
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # Binary classification
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification token
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token
        
        # Classification head
        x = self.head(x)
        return x

def get_vit(pretrained=False):
    """Returns an initialized ViT model for binary classification."""
    model = ViT(
        image_size=224,        # Required input size
        patch_size=16,         # Size of patches
        in_channels=3,         # RGB images
        embed_dim=768,         # Embedding dimension
        num_layers=12,         # Number of transformer blocks
        num_heads=12,          # Number of attention heads
        mlp_ratio=4.0,         # MLP hidden dimension ratio
        dropout=0.1            # Dropout rate
    )
    
    if pretrained:
        # You would typically load pretrained weights here
        # This is left as a placeholder
        pass
        
    return model

def setup_environment():
    """Setup testing environment"""
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Fore.YELLOW}🚀 Using device: {device}{Style.RESET_ALL}")

    # Create results directory
    os.makedirs(TestConfig.RESULTS_DIR, exist_ok=True)
    print(f"{Fore.GREEN}✅ Results directory created at {TestConfig.RESULTS_DIR}{Style.RESET_ALL}")

    return device

def load_data():
    """Load test dataset"""
    print(f"{Fore.CYAN}📂 Loading test dataset...{Style.RESET_ALL}")
    
    try:
        # First try to load dedicated test data if available
        if os.path.exists(TestConfig.TEST_DATA_PATH):
            X_test, y_test = torch.load(TestConfig.TEST_DATA_PATH)
            test_dataset = TensorDataset(X_test, y_test)
            print(f"{Fore.GREEN}✅ Dedicated test dataset loaded from {TestConfig.TEST_DATA_PATH}{Style.RESET_ALL}")
        else:
            # Otherwise use a portion of the training data for testing
            print(f"{Fore.YELLOW}→ No dedicated test data found. Using a portion of training data.{Style.RESET_ALL}")
            X, y = torch.load(TestConfig.DATA_PATH)
            dataset = TensorDataset(X, y)
            
            # Use 20% of data for testing (non-overlapping with training data)
            total_samples = len(dataset)
            train_size = int(0.8 * total_samples)
            test_indices = list(range(train_size, total_samples))
            
            # Create a subset for testing only
            X_test = X[test_indices]
            y_test = y[test_indices]
            test_dataset = TensorDataset(X_test, y_test)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=TestConfig.BATCH_SIZE,
            num_workers=TestConfig.NUM_WORKERS,
            pin_memory=True,
            shuffle=False  # No need to shuffle for testing
        )
        
        print(f"{Fore.GREEN}✅ Test dataset loaded - {len(test_dataset)} samples{Style.RESET_ALL}")
        print(f"   ├─ Input shape (X): {X_test.shape}")
        print(f"   └─ Labels shape (y): {y_test.shape}")
        
        # Calculate class distribution
        real_count = (y_test == 0).sum().item()
        fake_count = (y_test == 1).sum().item()
        print(f"   ├─ Real images: {real_count} ({real_count/len(y_test)*100:.1f}%)")
        print(f"   └─ Fake images: {fake_count} ({fake_count/len(y_test)*100:.1f}%)")
        
        return test_loader
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading dataset: {str(e)}{Style.RESET_ALL}")
        raise

def load_model(device):
    """Load the trained ViT model"""
    print(f"{Fore.CYAN}📥 Loading ViT model from {TestConfig.MODEL_PATH}...{Style.RESET_ALL}")
    
    try:
        # Create model architecture
        model = get_vit(pretrained=False).to(device)
        
        # Load saved weights
        model.load_state_dict(torch.load(TestConfig.MODEL_PATH, map_location=device))
        
        print(f"{Fore.GREEN}✅ ViT model loaded successfully!{Style.RESET_ALL}")
        return model
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading ViT model: {str(e)}{Style.RESET_ALL}")
        raise

def test_model(model, test_loader, device):
    """Test the model and calculate metrics"""
    print(f"\n{Fore.MAGENTA}🔍 Starting ViT model evaluation...{Style.RESET_ALL}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for ROC curve
    
    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc=f"{Fore.CYAN}Testing{Style.RESET_ALL}",
            ncols=100,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Style.RESET_ALL)
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            
            # Convert outputs to probabilities and predictions
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Store results
            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(labels.cpu().numpy().tolist() if hasattr(labels.cpu().numpy(), 'tolist') else [labels.cpu().numpy()])
            all_probs.extend(probs.tolist() if hasattr(probs, 'tolist') else [probs])
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    all_probs = np.array(all_probs).reshape(-1)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Store metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": float(roc_auc)
    }
    
    # Print metrics
    print(f"\n{Fore.GREEN}✨ ViT Test Results:{Style.RESET_ALL}")
    print(f"   ├─ Accuracy: {accuracy:.4f}")
    print(f"   ├─ Precision: {precision:.4f}")
    print(f"   ├─ Recall: {recall:.4f}")
    print(f"   ├─ F1 Score: {f1:.4f}")
    print(f"   └─ ROC AUC: {roc_auc:.4f}")
    
    return metrics, all_labels, all_preds, all_probs, fpr, tpr, roc_auc

def plot_confusion_matrix(conf_matrix, class_names=['Real', 'Fake']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ViT Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ViT ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_metrics_bar(metrics):
    """Plot bar chart of main metrics"""
    plt.figure(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metric_values = [
        metrics['accuracy'], 
        metrics['precision'], 
        metrics['recall'], 
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    bars = plt.bar(metric_names, metric_values, color=['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#9C27B0'])
    
    plt.ylim(0, 1.05)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('ViT Performance Metrics')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'metrics_bar.png'), dpi=300)
    plt.close()

def analyze_attention_maps(model, test_loader, device, num_samples=5):
    """Analyze and visualize attention maps from the ViT model"""
    print(f"\n{Fore.MAGENTA}🔍 Analyzing ViT attention patterns...{Style.RESET_ALL}")
    
    model.eval()
    
    # Get a few sample images
    sample_images = []
    sample_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            sample_batch_size = min(num_samples - len(sample_images), images.shape[0])
            sample_images.extend(images[:sample_batch_size])
            sample_labels.extend(labels[:sample_batch_size])
            
            if len(sample_images) >= num_samples:
                break
    
    # This is a placeholder for attention visualization
    # In a real implementation, you would need to modify the ViT model
    # to extract attention weights from each transformer block
    
    print(f"{Fore.YELLOW}→ Note: Attention map analysis requires model modification to extract attention weights.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}→ This is a placeholder for future implementation.{Style.RESET_ALL}")

def save_results(metrics):
    """Save test results to JSON file"""
    with open(TestConfig.METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"{Fore.GREEN}✅ Test metrics saved to {TestConfig.METRICS_PATH}{Style.RESET_ALL}")

def main():
    # Setup environment
    device = setup_environment()
    
    # Load test data
    test_loader = load_data()
    
    # Load model
    model = load_model(device)
    
    # Test model and get metrics
    metrics, all_labels, all_preds, all_probs, fpr, tpr, roc_auc = test_model(model, test_loader, device)
    
    # Visualize results
    print(f"\n{Fore.MAGENTA}📊 Generating visualizations...{Style.RESET_ALL}")
    plot_confusion_matrix(np.array(metrics['confusion_matrix']))
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_metrics_bar(metrics)
    
    # Optional: Analyze attention patterns
    # analyze_attention_maps(model, test_loader, device)
    
    # Save results
    save_results(metrics)
    
    print(f"\n{Fore.GREEN}✅ ViT testing completed!{Style.RESET_ALL}")
    print(f"Results saved to {TestConfig.RESULTS_DIR} directory")

if __name__ == "__main__":
    main()
