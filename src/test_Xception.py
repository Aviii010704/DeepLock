import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm
from colorama import Fore, Style
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from timm import create_model

class TestConfig:
    # Data paths
    DATA_PATH = "data/preprocessed_data_p1.pth"
    TEST_DATA_PATH = "data/preprocessed_data_test.pth"  # Optional separate test data
    
    # Model paths
    MODEL_DIR = "models/Final Models"
    MODEL_PATH = os.path.join(MODEL_DIR, "deeplock_xception.pth")
    
    # Output directories
    RESULTS_DIR = "test_results/xception"
    METRICS_PATH = os.path.join(RESULTS_DIR, "test_metrics.json")
    
    # Testing parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2

def get_xception(pretrained=False):
    """Load Xception model with custom classifier"""
    return create_model("xception", pretrained=pretrained, num_classes=1)

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
            X_test, y_test = torch.load(TestConfig.TEST_DATA_PATH, map_location='cpu')
            test_dataset = TensorDataset(X_test, y_test)
            print(f"{Fore.GREEN}✅ Dedicated test dataset loaded from {TestConfig.TEST_DATA_PATH}{Style.RESET_ALL}")
        else:
            # Otherwise use a portion of the training data for testing
            print(f"{Fore.YELLOW}→ No dedicated test data found. Using a portion of training data.{Style.RESET_ALL}")
            X, y = torch.load(TestConfig.DATA_PATH, map_location='cpu')
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
    """Load the trained Xception model"""
    print(f"{Fore.CYAN}📥 Loading Xception model from {TestConfig.MODEL_PATH}...{Style.RESET_ALL}")
    
    try:
        # Create model architecture
        model = get_xception(pretrained=False).to(device)
        
        # Load saved weights
        model.load_state_dict(torch.load(TestConfig.MODEL_PATH, map_location=device))
        
        print(f"{Fore.GREEN}✅ Xception model loaded successfully!{Style.RESET_ALL}")
        return model
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading Xception model: {str(e)}{Style.RESET_ALL}")
        raise

def test_model(model, test_loader, device):
    """Test the model and calculate metrics"""
    print(f"\n{Fore.MAGENTA}🔍 Starting Xception model evaluation...{Style.RESET_ALL}")
    
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
            outputs = model(images) / 10  # Apply same scaling as in training
            
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
    print(f"\n{Fore.GREEN}✨ Xception Test Results:{Style.RESET_ALL}")
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
    plt.title('Xception Confusion Matrix')
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
    plt.title('Xception ROC Curve')
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
    plt.title('Xception Performance Metrics')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'metrics_bar.png'), dpi=300)
    plt.close()

def analyze_facial_regions(model, device):
    """Optional: Analyze performance on different facial regions"""
    print(f"\n{Fore.MAGENTA}🔍 Analyzing model performance on facial regions...{Style.RESET_ALL}")
    
    # This is a placeholder for potential facial region analysis
    # Similar to what was done in the research papers
    # Would require additional data processing to isolate facial regions
    
    # Example regions that could be analyzed:
    regions = ['Eyes', 'Nose', 'Mouth', 'Full Face']
    accuracies = []
    
    # Placeholder for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(regions, [0.92, 0.88, 0.90, 0.94], color='skyblue')
    plt.ylim(0, 1.0)
    plt.xlabel('Facial Region')
    plt.ylabel('Detection Accuracy')
    plt.title('Xception Performance on Different Facial Regions')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(TestConfig.RESULTS_DIR, 'facial_regions.png'), dpi=300)
    plt.close()
    
    print(f"{Fore.YELLOW}→ Note: This is a placeholder. Implement actual facial region analysis if needed.{Style.RESET_ALL}")

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
    
    # Optional: Analyze performance on different facial regions
    # analyze_facial_regions(model, device)
    
    # Save results
    save_results(metrics)
    
    print(f"\n{Fore.GREEN}✅ Xception testing completed!{Style.RESET_ALL}")
    print(f"Results saved to {TestConfig.RESULTS_DIR} directory")

if __name__ == "__main__":
    main()
