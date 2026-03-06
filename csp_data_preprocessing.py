import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 50)
print("STEP 1: Loading Dataset")
print("=" * 50)

df = pd.read_csv("customer_support_tickets.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print("\n" + "=" * 50)
print("STEP 2: Basic Info & Data Types")
print("=" * 50)

print(df.info())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print("\n" + "=" * 50)
print("STEP 3: Dropping Unnecessary Columns")
print("=" * 50)

cols_to_drop = ['Ticket ID', 'Customer Name', 'Customer Email',
                'Ticket Description', 'Resolution']

df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")
print(f"Remaining columns: {list(df.columns)}")

print("\n" + "=" * 50)
print("STEP 4: Handling Missing Values")
print("=" * 50)

# Keep only rows where Customer Satisfaction Rating exists (target variable)
df = df[df['Customer Satisfaction Rating'].notna()]
print(f"Rows after dropping missing target: {len(df)}")

# Fill missing First Response Time and Time to Resolution with median
for col in ['First Response Time', 'Time to Resolution']:
    if df[col].dtype == 'object':
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Missing values after handling:\n{df.isnull().sum()}")

# ─────────────────────────────────────────
# 5. CONVERT DATE COLUMNS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Converting Date Columns")
print("=" * 50)

df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['Purchase Year']    = df['Date of Purchase'].dt.year
df['Purchase Month']   = df['Date of Purchase'].dt.month
df.drop(columns=['Date of Purchase'], inplace=True)

df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution']  = pd.to_datetime(df['Time to Resolution'], errors='coerce')

# Response duration in hours
df['Response Duration (hrs)'] = (
    df['Time to Resolution'] - df['First Response Time']
).dt.total_seconds() / 3600

df['Response Duration (hrs)'].fillna(df['Response Duration (hrs)'].median(), inplace=True)
df.drop(columns=['First Response Time', 'Time to Resolution'], inplace=True)

print("Date columns converted and duration feature created.")

# ─────────────────────────────────────────
# 6. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Encoding Categorical Variables")
print("=" * 50)

label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Label encoding done for all categorical columns.")

# ─────────────────────────────────────────
# 7. TARGET VARIABLE
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Target Variable Info")
print("=" * 50)

print(f"Target: Customer Satisfaction Rating")
print(f"Value counts:\n{df['Customer Satisfaction Rating'].value_counts().sort_index()}")

# Convert rating to integer
df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].astype(int)

# ─────────────────────────────────────────
# 8. SAVE CLEANED DATA
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8: Saving Cleaned Dataset")
print("=" * 50)

df.to_csv("cleaned_data.csv", index=False)
print(f"Cleaned data saved as 'cleaned_data.csv'")
print(f"Final shape: {df.shape}")
print(f"\nFinal columns:\n{list(df.columns)}")
print("\nData Preprocessing Complete!")
