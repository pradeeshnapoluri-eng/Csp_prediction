import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs("csp_models", exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD FINAL FEATURES
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Feature Dataset")
print("=" * 50)

df = pd.read_csv("csp_features_final.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

TARGET = 'Customer Satisfaction Rating'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\nFeatures : {X.shape}")
print(f"Target   : {y.shape}")
print(f"\nClass distribution:\n{y.value_counts().sort_index()}")

# ─────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Train / Test Split (70/30)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape}")
print(f"Testing set  : {X_test.shape}")

# ─────────────────────────────────────────
# 3. DEFINE MODELS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Defining Models")
print("=" * 50)

models = {
    "Logistic Regression"       : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"             : DecisionTreeClassifier(random_state=42),
    "Random Forest"             : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"         : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors"       : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine"    : SVC(kernel='rbf', random_state=42)
}

print(f"Total models to train: {len(models)}")

# ─────────────────────────────────────────
# 4. TRAIN & COMPARE MODELS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Training & Comparing All Models")
print("=" * 50)

results = []

for name, model in models.items():
    print(f"\nTraining: {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)

    # Cross validation (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    results.append({
        'Model'         : name,
        'Train Accuracy': round(train_acc * 100, 2),
        'Test Accuracy' : round(test_acc  * 100, 2),
        'CV Mean Acc'   : round(cv_mean   * 100, 2),
        'CV Std'        : round(cv_std    * 100, 2)
    })

    print(f"  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test Accuracy  : {test_acc  * 100:.2f}%")
    print(f"  CV Mean (5-fold): {cv_mean  * 100:.2f}% (+/- {cv_std * 100:.2f}%)")

# ─────────────────────────────────────────
# 5. RESULTS SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Model Comparison Summary")
print("=" * 50)

results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.index += 1
print(results_df.to_string())

# ─────────────────────────────────────────
# 6. BEST MODEL
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Selecting Best Model")
print("=" * 50)

best_model_name = "Random Forest"
best_model      = models[best_model_name]
print(f"Best Model : {best_model_name}")
print(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy']}%")

# ─────────────────────────────────────────
# 7. SAVE MODELS & RESULTS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Saving Models & Results")
print("=" * 50)

# Save best model
with open("csp_models/csp_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model saved: csp_models/csp_best_model.pkl")

# Save all models
for name, model in models.items():
    fname = name.lower().replace(" ", "_")
    with open(f"csp_models/csp_{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

print("All models saved in csp_models/ folder")

# Save test data for evaluation
X_test.to_csv("csp_X_test.csv", index=False)
y_test.to_csv("csp_y_test.csv", index=False)
X_train.to_csv("csp_X_train.csv", index=False)
y_train.to_csv("csp_y_train.csv", index=False)
print("Train/Test splits saved as CSV files")

# Save results
results_df.to_csv("csp_model_results.csv", index=False)
print("Model comparison results saved: csp_model_results.csv")

print("\n" + "=" * 50)
print("Model Building Complete!")
print("=" * 50)