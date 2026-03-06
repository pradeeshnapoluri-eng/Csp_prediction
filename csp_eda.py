import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder to save charts
os.makedirs("eda_charts", exist_ok=True)

print("=" * 50)
print("Loading Cleaned Dataset")
print("=" * 50)

df = pd.read_csv("cleaned_data.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

sns.set(style="whitegrid")

print("\nPlotting Chart 1: Satisfaction Distribution...")

plt.figure(figsize=(8, 5))
sns.countplot(x='Customer Satisfaction Rating', data=df, palette='Blues_d')
plt.title('Customer Satisfaction Rating Distribution')
plt.xlabel('Satisfaction Rating (1=Low, 5=High)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("eda_charts/01_satisfaction_distribution.png")
plt.close()
print("Saved: eda_charts/01_satisfaction_distribution.png")

print("\nPlotting Chart 2: Age Distribution...")

plt.figure(figsize=(10, 5))
sns.histplot(df['Customer Age'], bins=20, kde=True, color='salmon')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("eda_charts/02_age_distribution.png")
plt.close()
print("Saved: eda_charts/02_age_distribution.png")

print("\nPlotting Chart 3: Ticket Priority Distribution...")

priority_counts = df['Ticket Priority'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(priority_counts, labels=priority_counts.index,
        autopct='%1.1f%%', colors=sns.color_palette('pastel'),
        startangle=140)
plt.title('Ticket Priority Distribution')
plt.tight_layout()
plt.savefig("eda_charts/03_ticket_priority.png")
plt.close()
print("Saved: eda_charts/03_ticket_priority.png")

print("\nPlotting Chart 4: Ticket Type Distribution...")

ticket_type_counts = df['Ticket Type'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(ticket_type_counts, labels=ticket_type_counts.index,
        autopct='%1.1f%%', colors=sns.color_palette('Set2'),
        startangle=90)
plt.title('Ticket Type Distribution')
plt.tight_layout()
plt.savefig("eda_charts/04_ticket_type.png")
plt.close()
print("Saved: eda_charts/04_ticket_type.png")

print("\nPlotting Chart 5: Ticket Channel Distribution...")

channel_counts = df['Ticket Channel'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=channel_counts.index, y=channel_counts.values, palette='rocket')
plt.title('Ticket Channel Distribution')
plt.xlabel('Ticket Channel')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("eda_charts/05_ticket_channel.png")
plt.close()
print("Saved: eda_charts/05_ticket_channel.png")

print("\nPlotting Chart 6: Gender Distribution...")

gender_counts = df['Customer Gender'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=gender_counts.index,
        autopct='%1.1f%%', colors=sns.color_palette('Set3'),
        startangle=90)
plt.title('Customer Gender Distribution')
plt.tight_layout()
plt.savefig("eda_charts/06_gender_distribution.png")
plt.close()
print("Saved: eda_charts/06_gender_distribution.png")

# ─────────────────────────────────────────
# CHART 7: Satisfaction Rating by Ticket Type
# ─────────────────────────────────────────
print("\nPlotting Chart 7: Satisfaction by Ticket Type...")

plt.figure(figsize=(10, 5))
sns.boxplot(x='Ticket Type', y='Customer Satisfaction Rating',
            data=df, palette='coolwarm')
plt.title('Satisfaction Rating by Ticket Type')
plt.xlabel('Ticket Type (Encoded)')
plt.ylabel('Satisfaction Rating')
plt.tight_layout()
plt.savefig("eda_charts/07_satisfaction_by_ticket_type.png")
plt.close()
print("Saved: eda_charts/07_satisfaction_by_ticket_type.png")

# ─────────────────────────────────────────
# CHART 8: Satisfaction Rating by Ticket Priority
# ─────────────────────────────────────────
print("\nPlotting Chart 8: Satisfaction by Priority...")

plt.figure(figsize=(9, 5))
sns.boxplot(x='Ticket Priority', y='Customer Satisfaction Rating',
            data=df, palette='Set1')
plt.title('Satisfaction Rating by Ticket Priority')
plt.xlabel('Ticket Priority (Encoded)')
plt.ylabel('Satisfaction Rating')
plt.tight_layout()
plt.savefig("eda_charts/08_satisfaction_by_priority.png")
plt.close()
print("Saved: eda_charts/08_satisfaction_by_priority.png")

# ─────────────────────────────────────────
# CHART 9: Correlation Heatmap
# ─────────────────────────────────────────
print("\nPlotting Chart 9: Correlation Heatmap...")

plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("eda_charts/09_correlation_heatmap.png")
plt.close()
print("Saved: eda_charts/09_correlation_heatmap.png")

# ─────────────────────────────────────────
# CHART 10: Response Duration vs Satisfaction
# ─────────────────────────────────────────
print("\nPlotting Chart 10: Response Duration vs Satisfaction...")

plt.figure(figsize=(9, 5))
sns.boxplot(x='Customer Satisfaction Rating', y='Response Duration (hrs)',
            data=df, palette='viridis')
plt.title('Response Duration vs Customer Satisfaction')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Response Duration (hrs)')
plt.tight_layout()
plt.savefig("eda_charts/10_response_duration_vs_satisfaction.png")
plt.close()
print("Saved: eda_charts/10_response_duration_vs_satisfaction.png")

# ─────────────────────────────────────────
# PRINT SUMMARY STATS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("EDA Complete! All 10 charts saved in 'eda_charts/' folder.")
print("=" * 50)
