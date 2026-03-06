import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("STEP 1: Loading All Results")
print("=" * 50)

df_cleaned   = pd.read_csv("csp_cleaned_data.csv")
df_features  = pd.read_csv("csp_features_final.csv")
df_models    = pd.read_csv("csp_model_results.csv")
df_eval      = pd.read_csv("csp_evaluation_summary.csv")
X_test       = pd.read_csv("csp_X_test.csv")
y_test       = pd.read_csv("csp_y_test.csv").squeeze()

with open("csp_models/csp_best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

y_pred = best_model.predict(X_test)

print("All files loaded successfully.")

print("\n" + "=" * 50)
print("STEP 2: Dataset Summary Sheet")
print("=" * 50)

dataset_summary = pd.DataFrame({
    'Metric': [
        'Total Rows (Raw)',
        'Total Columns (Raw)',
        'Rows After Cleaning',
        'Columns After Cleaning',
        'Missing Values (Raw)',
        'Duplicate Rows'
    ],
    'Value': [
        8469,
        17,
        len(df_cleaned),
        df_cleaned.shape[1],
        5700,
        0
    ]
})
print(dataset_summary.to_string(index=False))

print("\n" + "=" * 50)
print("STEP 3: Satisfaction Distribution Sheet")
print("=" * 50)

TARGET = 'Customer Satisfaction Rating'
sat_dist = df_cleaned[TARGET].value_counts().sort_index().reset_index()
sat_dist.columns = ['Rating', 'Count']
sat_dist['Percentage'] = (sat_dist['Count'] / sat_dist['Count'].sum() * 100).round(2)
print(sat_dist.to_string(index=False))

print("\n" + "=" * 50)
print("STEP 4: Model Results Sheet")
print("=" * 50)

print(df_models.to_string(index=False))

print("\n" + "=" * 50)
print("STEP 5: Classification Report Sheet")
print("=" * 50)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()
report_df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
report_df = report_df.round(2)
print(report_df.to_string(index=False))

print("\n" + "=" * 50)
print("STEP 6: Feature Importance Sheet")
print("=" * 50)

if hasattr(best_model, 'feature_importances_'):
    fi_df = pd.DataFrame({
        'Feature'   : X_test.columns,
        'Importance': best_model.feature_importances_.round(4)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    print(fi_df.to_string(index=False))
else:
    fi_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': ['N/A'] * len(X_test.columns)})
    print("Feature importance not available for this model.")

print("\n" + "=" * 50)
print("STEP 7: Saving csp_report.csv (Combined)")
print("=" * 50)

dataset_summary.to_csv("csp_report_dataset_summary.csv",    index=False)
sat_dist.to_csv("csp_report_satisfaction_dist.csv",         index=False)
df_models.to_csv("csp_report_model_results.csv",            index=False)
report_df.to_csv("csp_report_classification_report.csv",    index=False)
fi_df.to_csv("csp_report_feature_importance.csv",           index=False)

combined_rows = []

combined_rows.append(pd.DataFrame([['=== DATASET SUMMARY ===', '']],
                     columns=['Metric', 'Value']))
combined_rows.append(dataset_summary)
combined_rows.append(pd.DataFrame([['', '']], columns=['Metric', 'Value']))

combined_rows.append(pd.DataFrame([['=== SATISFACTION DISTRIBUTION ===', '', '']],
                     columns=['Rating', 'Count', 'Percentage']))
combined_rows.append(sat_dist)

report_combined = pd.concat(combined_rows, ignore_index=True)
report_combined.to_csv("csp_report.csv", index=False)

print("Saved: csp_report.csv")
print("Saved: csp_report_dataset_summary.csv")
print("Saved: csp_report_satisfaction_dist.csv")
print("Saved: csp_report_model_results.csv")
print("Saved: csp_report_classification_report.csv")
print("Saved: csp_report_feature_importance.csv")

print("\n" + "=" * 50)
print("Report Generation Complete!")
print("=" * 50)
