import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("fe_charts", exist_ok=True)
print("=" * 50)
print("STEP 1: Loading Cleaned Dataset")
print("=" * 50)

df = pd.read_csv("cleaned_data.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n" + "=" * 50)
print("STEP 2: Defining Features and Target")
print("=" * 50)

TARGET = 'Customer Satisfaction Rating'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"Feature columns: {list(X.columns)}")
print("\n" + "=" * 50)
print("STEP 3: Creating New Features")
print("=" * 50)

# Age group binning
bins   = [0, 25, 35, 50, 65, 100]
labels = [1, 2, 3, 4, 5]
X = X.copy()
X['Age Group'] = pd.cut(X['Customer Age'], bins=bins,
                         labels=labels).astype(int)
print("New feature created: Age Group (1=18-25, 2=26-35, 3=36-50, 4=51-65, 5=65+)")

# Response Duration bucket
X['Duration Bucket'] = pd.cut(X['Response Duration (hrs)'],
                               bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
print("New feature created: Duration Bucket (1=fastest, 5=slowest)")

# Purchase recency (how old is the product)
X['Purchase Recency'] = 2024 - X['Purchase Year']
print("New feature created: Purchase Recency (years since purchase)")

print(f"\nUpdated feature shape: {X.shape}")
print(f"All features: {list(X.columns)}")
print("\n" + "=" * 50)
print("STEP 4: Feature Scaling")
print("=" * 50)

# StandardScaler for continuous features
scale_cols = ['Customer Age', 'Response Duration (hrs)', 'Purchase Recency']
scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])
print(f"StandardScaler applied to: {scale_cols}")

# MinMaxScaler for month and year
minmax_cols = ['Purchase Year', 'Purchase Month']
mm_scaler = MinMaxScaler()
X[minmax_cols] = mm_scaler.fit_transform(X[minmax_cols])
print(f"MinMaxScaler applied to  : {minmax_cols}")

print("\nSample after scaling:")
print(X.head(3))
print("\n" + "=" * 50)
print("STEP 5: Feature Selection (SelectKBest - ANOVA F-test)")
print("=" * 50)

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({
    'Feature' : X.columns,
    'F-Score' : selector.scores_,
    'P-Value' : selector.pvalues_
}).sort_values('F-Score', ascending=False).reset_index(drop=True)

print("\nFeature Scores (sorted by F-Score):")
print(feature_scores.to_string(index=False))

# Select top 8 features
top_features = feature_scores['Feature'].head(8).tolist()
print(f"\nTop 8 Selected Features: {top_features}")
print("\n" + "=" * 50)
print("STEP 6: Plotting Feature Importance")
print("=" * 50)

plt.figure(figsize=(10, 6))
sns.barplot(x='F-Score', y='Feature', data=feature_scores, palette='magma')
plt.title('Feature Importance (ANOVA F-Score)')
plt.xlabel('F-Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("fe_charts/feature_importance.png")
plt.close()
print("Saved: fe_charts/feature_importance.png")
print("\n" + "=" * 50)
print("STEP 7: Saving Final Feature Dataset")
print("=" * 50)

X_selected = X[top_features].copy()
X_selected[TARGET] = y.values

X_selected.to_csv("features_final.csv", index=False)
print(f"Saved: features_final.csv")
print(f"Shape: {X_selected.shape}")
print(f"Columns: {list(X_selected.columns)}")

print("\n" + "=" * 50)
print("Feature Engineering Complete!")
print("=" * 50)