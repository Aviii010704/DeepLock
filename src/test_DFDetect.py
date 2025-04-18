# test.py
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
from model import DFDetectV2, get_dfdetect, DeepFakeDetector

class TestConfig:
    # Data paths
    DATA_PATH = "data/preprocessed_data_p1.pth"  # Use the same data path as in train.py
    TEST_DATA_PATH = "data/preprocessed_data_test.pth"  # Optional separate test data
    
    # Model paths
    MODEL_DIR = "models/Final Models"
    MODEL_PATH = os.path.join(MODEL_DIR, "deeplock_dfdetect_v2.pth")
    
    # Output directories
    RESULTS_DIR = "test_results"
    METRICS_PATH = os.path.join(RESULTS_DIR, "test_metrics.json")
    
    # Testing parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    USE_DFDETECTV2 = False  # Set to False to use ResNet-based model, which appears to be what was trained

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
    """Load the trained model with auto-detection of model type"""
    print(f"{Fore.CYAN}📥 Loading model from {TestConfig.MODEL_PATH}...{Style.RESET_ALL}")
    
    try:
        # Try to determine model type from state dict
        state_dict = torch.load(TestConfig.MODEL_PATH, map_location=device)
        
        # Check keys to determine model type
        has_resnet_keys = any(key.startswith(('layer', 'conv1', 'bn1')) for key in state_dict.keys())
        has_vit_keys = any(key.startswith('backbone') for key in state_dict.keys())
        
        if has_resnet_keys:
            print(f"{Fore.YELLOW}→ Detected ResNet-based model from state dict keys{Style.RESET_ALL}")
            model = get_dfdetect(pretrained=False).to(device)
            model.load_state_dict(state_dict)
        elif has_vit_keys:
            print(f"{Fore.YELLOW}→ Detected ViT-based model (DFDetectV2) from state dict keys{Style.RESET_ALL}")
            model = DFDetectV2().to(device)
            model.load_state_dict(state_dict)
        else:
            # If can't determine from keys, use config setting
            if TestConfig.USE_DFDETECTV2:
                print(f"{Fore.YELLOW}→ Using advanced DFDetectV2 (ViT + Frequency Attention) based on config{Style.RESET_ALL}")
                model = DFDetectV2().to(device)
            else:
                print(f"{Fore.YELLOW}→ Using ResNet50-based DFDetect based on config{Style.RESET_ALL}")
                model = get_dfdetect(pretrained=False).to(device)
            model.load_state_dict(state_dict)
        
        print(f"{Fore.GREEN}✅ Model loaded successfully!{Style.RESET_ALL}")
        return model
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error when auto-detecting model type: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}→ Trying alternative model architecture...{Style.RESET_ALL}")
        
        try:
            # Create a DeepFakeDetector instance (which contains the ResNet model)
            model = DeepFakeDetector(pretrained=False).to(device)
            model.base_model.load_state_dict(torch.load(TestConfig.MODEL_PATH, map_location=device))
            print(f"{Fore.GREEN}✅ Model loaded successfully with DeepFakeDetector wrapper!{Style.RESET_ALL}")
            return model
        except Exception as e2:
            print(f"{Fore.RED}❌ Error loading model with wrapper: {str(e2)}{Style.RESET_ALL}")
            
            # Final fallback: recreate the ResNet model but with a direct architecture that matches the saved weights
            try:
                print(f"{Fore.YELLOW}→ Final attempt: Loading using ResNet50 base model...{Style.RESET_ALL}")
                model = model.resnet50(pretrained=False)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 1)
                )
                model.to(device)
                model.load_state_dict(torch.load(TestConfig.MODEL_PATH, map_location=device))
                print(f"{Fore.GREEN}✅ Model loaded with direct ResNet50 architecture!{Style.RESET_ALL}")
                return model
            except Exception as e3:
                print(f"{Fore.RED}❌ All attempts to load model failed: {str(e3)}{Style.RESET_ALL}")
                raise

def test_model(model, test_loader, device):
    """Test the model and calculate metrics"""
    print(f"\n{Fore.MAGENTA}🔍 Starting model evaluation...{Style.RESET_ALL}")
    
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
            
            # Handle both wrapped and direct models
            if hasattr(model, 'predict'):
                outputs = model.predict(images)
                probs = torch.sigmoid(model(images)).squeeze().cpu().numpy()
            else:
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Handle different output formats
            if isinstance(outputs, torch.Tensor) and outputs.shape[-1] == 1:
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().squeeze()
            elif isinstance(outputs, torch.Tensor):
                preds = (outputs > 0).cpu().numpy().astype(int).squeeze()
            else:
                preds = outputs.astype(int)
            
            all_preds.extend(preds if isinstance(preds, list) else preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(labels.cpu().numpy().tolist() if hasattr(labels.cpu().numpy(), 'tolist') else [labels.cpu().numpy()])
            all_probs.extend(probs if isinstance(probs, list) else probs.tolist() if hasattr(probs, 'tolist') else [probs])
    
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
    print(f"\n{Fore.GREEN}✨ Test Results:{Style.RESET_ALL}")
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
    plt.title('Confusion Matrix')
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
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
    plt.title('Performance Metrics')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'metrics_bar.png'), dpi=300)
    plt.close()

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
    
    # Save results
    save_results(metrics)
    
    print(f"\n{Fore.GREEN}✅ Testing completed!{Style.RESET_ALL}")
    print(f"Results saved to {TestConfig.RESULTS_DIR} directory")

if __name__ == "__main__":
    main()