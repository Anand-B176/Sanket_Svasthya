# Evaluation Script - Sanket-Svasthya
"""
Model evaluation with comprehensive metrics.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model(model_path: str, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   classes: np.ndarray,
                   output_dir: str = "../results"):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model
        X_test: Test features
        y_test: Test labels
        classes: Class names
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    print(f"Precision (macro): {precision:.2%}")
    print(f"Recall (macro): {recall:.2%}")
    print(f"F1-Score (macro): {f1:.2%}")
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_test, y_pred, target_names=classes)
    print(report)
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # Save metrics JSON
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_score_macro': float(f1),
        'num_test_samples': len(y_test),
        'num_classes': len(classes)
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_dir}/metrics.json")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate sign language model")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output", type=str, default="../results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load test data
    X_test = np.load(os.path.join(args.data, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data, 'y_test.npy'))
    classes = np.load(os.path.join(args.data, 'classes.npy'), allow_pickle=True)
    
    # Evaluate
    evaluate_model(args.model, X_test, y_test, classes, args.output)


if __name__ == "__main__":
    main()
