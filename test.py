import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1️ Load test dataset
TEST_PATH = "Dataset/test.csv"
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"❌ Test file not found: {TEST_PATH}")

test_data = pd.read_csv(TEST_PATH)
print("✅ Test data loaded successfully. Shape:", test_data.shape)

# 2️ Split features and target
X_test = test_data.drop("is_promoted", axis=1)
y_test = test_data["is_promoted"]

# 3️ Load trained models
MODEL_DIR = "ml_models"
rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
xgb_model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")

if not os.path.exists(rf_model_path) or not os.path.exists(xgb_model_path):
    raise FileNotFoundError("❌ Trained models not found. Please run train_model.py first.")

rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)
print("✅ Models loaded successfully!")

# 4️ Define evaluation function
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    print(f"\n===== {name} Evaluation on Test Data =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("\nClassification Report:\n", classification_report(y, y_pred))

    # Blue-white gradient colormap (white → navy blue)
    navy_blue_cmap = LinearSegmentedColormap.from_list("white_to_navy", ["#FFFFFF", "#000080"])
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap=navy_blue_cmap)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # Return metrics, predictions, and probabilities
    return {
        "name": name,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "predictions": pd.DataFrame({
            "Actual": y,
            f"{name}_Predicted": y_pred,
            f"{name}_Confidence": y_prob
        })
    }

# 5️ Evaluate both models
rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

# 6️ Merge results for predictions with confidence
merged = pd.concat([
    rf_results["predictions"],
    xgb_results["predictions"][[f"XGBoost_Predicted", f"XGBoost_Confidence"]]
], axis=1)

# Print each prediction with confidence
print("\n===== Test Set Predictions with Confidence =====")
print(merged.head(20))  # show first 20 for readability

# Save to CSV
merged.to_csv("Dataset/test_predictions_with_confidence.csv", index=False)
print("\n✅ Predictions with confidence saved as 'Dataset/test_predictions_with_confidence.csv'")

# 7️Model Comparison Summary
comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_results["accuracy"], xgb_results["accuracy"]],
    "F1-score": [rf_results["f1"], xgb_results["f1"]],
    "ROC-AUC": [rf_results["roc_auc"], xgb_results["roc_auc"]]
})
print("\n✅ Model Comparison:\n", comparison)

# Plot ROC curves together
plt.figure(figsize=(8, 6))

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_results["y_prob"])
rf_auc = auc(rf_fpr, rf_tpr)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label=f'Random Forest (AUC = {rf_auc:.4f})')

# XGBoost ROC
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_results["y_prob"])
xgb_auc = auc(xgb_fpr, xgb_tpr)
plt.plot(xgb_fpr, xgb_tpr, color='green', lw=2, label=f'XGBoost (AUC = {xgb_auc:.4f})')

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest vs XGBoost')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
