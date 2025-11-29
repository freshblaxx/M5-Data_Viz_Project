# ============================
# 1. Imports and setup
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots a bit nicer
plt.style.use("default")
sns.set()

# ============================
# 2. Load data
# ============================
file_path = "mental_health_social_media_dataset.csv"   # change if needed
df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())

# ============================
# 3. Basic info and data types
# ============================
print("\nInfo:")
print(df.info())

print("\nSummary of numeric columns:")
print(df.describe())

print("\nSummary of non numeric columns:")
print(df.describe(include="object"))

# Convert date column to datetime if needed
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print("\nCheck date column:")
    print(df["date"].describe())

# ============================
# 4. Missing values and duplicates
# ============================
print("\nMissing values per column:")
print(df.isna().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

# Optionally drop duplicates
# df = df.drop_duplicates()

# ============================
# 5. Categorical distributions
# ============================
categorical_cols = ["gender", "platform", "mental_state"]

for col in categorical_cols:
    if col in df.columns:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

        plt.figure()
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

# ============================
# 6. Numeric distributions
# ============================
numeric_cols = [
    "age",
    "daily_screen_time_min",
    "social_media_time_min",
    "negative_interactions_count",
    "positive_interactions_count",
    "sleep_hours",
    "physical_activity_min",
    "anxiety_level",
    "stress_level",
    "mood_level",
]

numeric_cols = [c for c in numeric_cols if c in df.columns]

# Histograms
for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Boxplots
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# ============================
# 7. Relationships between variables
# ============================

# Correlation matrix
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix of numeric features")
plt.tight_layout()
plt.show()

# Selected scatter plots
def safe_scatter(x, y):
    if x in df.columns and y in df.columns:
        plt.figure()
        sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
        plt.title(f"{y} versus {x}")
        plt.tight_layout()
        plt.show()

safe_scatter("social_media_time_min", "mood_level")
safe_scatter("social_media_time_min", "anxiety_level")
safe_scatter("social_media_time_min", "stress_level")
safe_scatter("sleep_hours", "stress_level")
safe_scatter("sleep_hours", "mood_level")

# ============================
# 8. Grouped analysis
# ============================

# By mental_state
if "mental_state" in df.columns:
    print("\nMean values by mental_state:")
    print(df.groupby("mental_state")[numeric_cols].mean().round(2))

    plt.figure(figsize=(10, 5))
    df.groupby("mental_state")["stress_level"].mean().plot(kind="bar")
    plt.title("Average stress level by mental state")
    plt.xlabel("Mental state")
    plt.ylabel("Average stress level")
    plt.tight_layout()
    plt.show()

# By platform
if "platform" in df.columns:
    print("\nMean values by platform:")
    print(df.groupby("platform")[numeric_cols].mean().round(2))

    plt.figure(figsize=(10, 5))
    df.groupby("platform")["daily_screen_time_min"].mean().plot(kind="bar")
    plt.title("Average daily screen time by platform")
    plt.xlabel("Platform")
    plt.ylabel("Average screen time (min)")
    plt.tight_layout()
    plt.show()

# By gender
if "gender" in df.columns:
    print("\nMean values by gender:")
    print(df.groupby("gender")[numeric_cols].mean().round(2))

    plt.figure(figsize=(10, 5))
    df.groupby("gender")["stress_level"].mean().plot(kind="bar")
    plt.title("Average stress level by gender")
    plt.xlabel("Gender")
    plt.ylabel("Average stress level")
    plt.tight_layout()
    plt.show()

# ============================
# 9. Time based exploration (if date exists)
# ============================
if "date" in df.columns:
    # Sort by date
    df = df.sort_values("date")

    # Resample per week if multiple days exist
    time_cols = [c for c in ["stress_level", "anxiety_level", "mood_level"] if c in df.columns]

    if len(time_cols) > 0:
        weekly = (
            df.set_index("date")
              .resample("W")[time_cols]
              .mean()
        )

        print("\nWeekly average mental health scores:")
        print(weekly.head())

        plt.figure(figsize=(10, 5))
        for c in time_cols:
            plt.plot(weekly.index, weekly[c], label=c)
        plt.title("Weekly average mental health scores over time")
        plt.xlabel("Date")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.show()