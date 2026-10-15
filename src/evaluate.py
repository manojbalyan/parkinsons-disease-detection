import json
import sys
from pathlib import Path

# Add project root to sys.path so imports work cleanly from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.dataset import load_and_preprocess_data
from src.model import ParkinsonNet

def evaluate_model(
    model_path: str = "models/best_parkinsons_model.pth",
    data_path: str = "data/raw/parkinsons.data",
    reports_dir: str = "reports"
):
    """
    Evaluates the saved best model on the unseen test set (39 samples).
    Calculates:
      - Accuracy: Overall correct predictions percentage.
      - Precision: Out of all predicted Parkinson's, how many actually had it.
      - Recall (Sensitivity): Out of all true Parkinson's patients, how many we caught.
      - Specificity: Out of all healthy controls, how many we correctly called healthy.
      - F1-Score: Harmonic mean of precision and recall.
      - ROC-AUC: Area under the ROC curve.
    Saves visual plots:
      - reports/confusion_matrix.png
      - reports/roc_curve.png
      - reports/evaluation_metrics.json
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the unseen 20% test dataset
    _, X_test, _, y_test, scaler, feature_names = load_and_preprocess_data(
        data_path=data_path,
        test_size=0.2,
        random_state=42,
        save_artifacts=False
    )
    
    # 2. Load the trained neural network checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model = ParkinsonNet(input_dim=len(feature_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Model Inference on Test Set
    with torch.no_grad():
        test_tensors = torch.tensor(X_test, dtype=torch.float32)
        probs = model(test_tensors, return_logits=False).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    # 4. Calculate Diagnostic Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    # Specificity = True Negatives / (True Negatives + False Positives)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "test_sample_count": len(y_test)
    }
    
    # Save metrics JSON file
    with open(reports_path / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 55)
    print("      PARKINSON'S DISEASE DETECTION - TEST EVALUATION    ")
    print("=" * 55)
    print(f"  Test Samples Evaluated : {len(y_test)}")
    print(f"  Accuracy               : {acc * 100:.2f}%")
    print(f"  Precision              : {prec * 100:.2f}%")
    print(f"  Recall (Sensitivity)   : {rec * 100:.2f}%")
    print(f"  Specificity            : {specificity * 100:.2f}%")
    print(f"  F1-Score               : {f1:.4f}")
    print(f"  ROC-AUC Score          : {roc_auc:.4f}")
    print("=" * 55)
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=["Healthy (0)", "Parkinson's (1)"]))
    
    # 5. Plot Confusion Matrix (Textbook TP/FP/FN/TN Quadrant Format)
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(8, 8.5), dpi=200)
    bg_color = '#FDF6ED'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(5.0, 10.15, 'Confusion Matrix in Machine Learning', 
            fontsize=20, fontweight='bold', ha='center', va='center', color='#1A202C')
    ax.text(5.0, 9.6, f"(Parkinson's Disease Detection — {len(y_test)} Unseen Test Samples)", 
            fontsize=11, fontstyle='italic', ha='center', va='center', color='#556987')

    purple_bg = '#C5CEFA'
    pink_bg = '#F8B6C1'
    header_bg = '#FFFFFF'
    border_color = '#1A1A1A'
    lw_box = 2.0

    # Column Headers (Top)
    ax.add_patch(patches.Rectangle((3.2, 7.8), 3.0, 0.9, facecolor=header_bg, edgecolor=border_color, lw=lw_box))
    ax.text(4.7, 8.25, 'Positive', fontsize=17, fontweight='bold', fontstyle='italic', ha='center', va='center', color='#111111')
    ax.add_patch(patches.Rectangle((6.2, 7.8), 3.0, 0.9, facecolor=header_bg, edgecolor=border_color, lw=lw_box))
    ax.text(7.7, 8.25, 'Negative', fontsize=17, fontweight='bold', fontstyle='italic', ha='center', va='center', color='#111111')

    # Row Headers (Left)
    ax.add_patch(patches.Rectangle((2.1, 4.8), 1.1, 3.0, facecolor=header_bg, edgecolor=border_color, lw=lw_box))
    ax.text(2.65, 6.3, 'Positive', fontsize=17, fontweight='bold', fontstyle='italic', ha='center', va='center', rotation=90, color='#111111')
    ax.add_patch(patches.Rectangle((2.1, 1.8), 1.1, 3.0, facecolor=header_bg, edgecolor=border_color, lw=lw_box))
    ax.text(2.65, 3.3, 'Negative', fontsize=17, fontweight='bold', fontstyle='italic', ha='center', va='center', rotation=90, color='#111111')

    # Top-Left: TP
    ax.add_patch(patches.Rectangle((3.2, 4.8), 3.0, 3.0, facecolor=purple_bg, edgecolor=border_color, lw=lw_box))
    ax.text(4.7, 6.6, 'TP', fontsize=28, fontweight='bold', ha='center', va='center', color='#111111')
    ax.text(4.7, 5.5, f'= {tp}', fontsize=34, fontweight='bold', ha='center', va='center', color='#0A2647')
    ax.text(4.7, 4.95, '(True Positive)', fontsize=11, fontweight='bold', ha='center', va='center', color='#2C3E50')

    # Top-Right: FP
    ax.add_patch(patches.Rectangle((6.2, 4.8), 3.0, 3.0, facecolor=pink_bg, edgecolor=border_color, lw=lw_box))
    ax.text(7.7, 6.6, 'FP', fontsize=28, fontweight='bold', ha='center', va='center', color='#111111')
    ax.text(7.7, 5.5, f'= {fp}', fontsize=34, fontweight='bold', ha='center', va='center', color='#991B1B')
    ax.text(7.7, 4.95, '(False Positive)', fontsize=11, fontweight='bold', ha='center', va='center', color='#991B1B')

    # Bottom-Left: FN
    ax.add_patch(patches.Rectangle((3.2, 1.8), 3.0, 3.0, facecolor=pink_bg, edgecolor=border_color, lw=lw_box))
    ax.text(4.7, 3.6, 'FN', fontsize=28, fontweight='bold', ha='center', va='center', color='#111111')
    ax.text(4.7, 2.5, f'= {fn}', fontsize=34, fontweight='bold', ha='center', va='center', color='#991B1B')
    ax.text(4.7, 1.95, '(False Negative)', fontsize=11, fontweight='bold', ha='center', va='center', color='#991B1B')

    # Bottom-Right: TN
    ax.add_patch(patches.Rectangle((6.2, 1.8), 3.0, 3.0, facecolor=purple_bg, edgecolor=border_color, lw=lw_box))
    ax.text(7.7, 3.6, 'TN', fontsize=28, fontweight='bold', ha='center', va='center', color='#111111')
    ax.text(7.7, 2.5, f'= {tn}', fontsize=34, fontweight='bold', ha='center', va='center', color='#0A2647')
    ax.text(7.7, 1.95, '(True Negative)', fontsize=11, fontweight='bold', ha='center', va='center', color='#2C3E50')

    # Footer metrics box
    ax.add_patch(patches.Rectangle((1.5, 0.45), 7.2, 0.9, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.2))
    ax.text(5.1, 1.0, f'Accuracy = (TP + TN) / Total  =  ({tp} + {tn}) / {len(y_test)}  =  {acc * 100:.1f}%', fontsize=11.5, fontweight='bold', ha='center', color='#0F172A')
    ax.text(5.1, 0.65, f'Sensitivity (Recall) = {rec * 100:.1f}%   |   Precision = {prec * 100:.1f}%', fontsize=9.5, ha='center', color='#475569')

    cm_path = reports_path / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=200, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # 6. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = reports_path / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}\n")

if __name__ == "__main__":
    evaluate_model()
