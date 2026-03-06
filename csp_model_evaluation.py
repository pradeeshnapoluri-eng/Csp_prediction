import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

os.makedirs("csp_eval_charts", exist_ok=True)

print("=" * 50)
print("STEP 1: Loading Data & Best Model")
print("=" * 50)

X_test  = pd.read_csv("csp_X_test.csv")
y_test  = pd.read_csv("csp_y_test.csv").squeeze()
X_train = pd.read_csv("csp_X_train.csv")
y_train = pd.read_csv("csp_y_train.csv").squeeze()

with open("csp_models/csp_best_model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"Model loaded : {type(model).__name__}")
print(f"X_test shape : {X_test.shape}")

y_pred       = model.predict(X_test)
train_acc    = accuracy_score(y_train, model.predict(X_train))
test_acc     = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("STEP 2: Accuracy")
print("=" * 50)
print(f"Train Accuracy : {train_acc * 100:.2f}%")
print(f"Test Accuracy  : {test_acc  * 100:.2f}%")

print("\n" + "=" * 50)
print("STEP 3: Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred,
      target_names=['Rating 1','Rating 2','Rating 3','Rating 4','Rating 5']))

print("\n" + "=" * 50)
print("STEP 4: Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[1,2,3,4,5],
            yticklabels=[1,2,3,4,5])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("csp_eval_charts/01_confusion_matrix.png")
plt.close()
print("Saved: csp_eval_charts/01_confusion_matrix.png")

print("\n" + "=" * 50)
print("STEP 5: Feature Importance")
print("=" * 50)

if hasattr(model, 'feature_importances_'):
    fi = pd.Series(model.feature_importances_, index=X_test.columns)
    fi = fi.sort_values(ascending=False)
    print(fi)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fi.values, y=fi.index, palette='magma')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig("csp_eval_charts/02_feature_importance.png")
    plt.close()
    print("Saved: csp_eval_charts/02_feature_importance.png")
else:
    print("Feature importance not available for this model type.")

print("\n" + "=" * 50)
print("STEP 6: Model Comparison Chart")
print("=" * 50)

results_df = pd.read_csv("csp_model_results.csv")

plt.figure(figsize=(10, 6))
x = range(len(results_df))
plt.bar(x, results_df['Train Accuracy'], width=0.4,
        label='Train Accuracy', align='center', color='steelblue')
plt.bar([i + 0.4 for i in x], results_df['Test Accuracy'], width=0.4,
        label='Test Accuracy', align='center', color='salmon')
plt.xticks([i + 0.2 for i in x],
           results_df['Model'], rotation=30, ha='right')
plt.title('Model Comparison - Train vs Test Accuracy')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig("csp_eval_charts/03_model_comparison.png")
plt.close()
print("Saved: csp_eval_charts/03_model_comparison.png")

print("\n" + "=" * 50)
print("STEP 7: Saving Evaluation Summary")
print("=" * 50)

summary = pd.DataFrame({
    'Metric' : ['Train Accuracy', 'Test Accuracy'],
    'Value'  : [f"{train_acc*100:.2f}%", f"{test_acc*100:.2f}%"]
})
summary.to_csv("csp_evaluation_summary.csv", index=False)
print("Saved: csp_evaluation_summary.csv")

print("\n" + "=" * 50)
print("Model Evaluation Complete!")
print("=" * 50)
