import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load & Inspect Data
df = pd.read_csv("initial_dataset.csv")
print("Columns:", df.columns)
print(df.head())

# Replace nulls in Certification column with "None"
df['Certification'] = df['Certification'].fillna('None')
df_clean = df_clean[df_clean["Experience"] <= (df_clean["Age"] - 18)]
df_clean = df_clean[df_clean["Age"] >= 18]
print("Shape after Age/Experience cleaning:", df_clean.shape)
df_clean = df_clean.reset_index(drop=True)  # Reset DataFrame index, drop old index column
df_clean['S.NO'] = df_clean.index + 1      # Assign new consecutive numbers starting from 1
df_clean.to_csv('corrected_expanded_job_roles.csv', index=False)

# 2. Summary Statistics and Missing Value Analysis
print("Shape:", df.shape)
print(df.info())
print("Missing values per column:\n", df.isnull().sum())
print("Statistical summary:\n", df.describe(include='all'))

# 3. Univariate Analysis
numerical_cols = ['Age', 'Experience']
categorical_cols = ['Degree', 'Domain', 'JobRole']

for col in numerical_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"{col} distribution")
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(8,3))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"{col} countplot")
    plt.show()

# 4. Bivariate Analysis
# Scatterplot: Age vs. Experience
plt.figure(figsize=(6,4))
sns.scatterplot(x="Age", y="Experience", data=df)
plt.title("Age vs. Experience")
plt.show()

# Boxplot: Experience by Degree
plt.figure(figsize=(10,4))
sns.boxplot(x="Degree", y="Experience", data=df)
plt.title("Experience by Degree")
plt.xticks(rotation=45)
plt.show()

# Groupby: Mean Experience per Degree
print(df.groupby("Degree")["Experience"].mean())

# 5. Correlation Analysis
corr = df[numerical_cols].corr()
print("Correlation matrix:\n", corr)
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 6. Pairwise Relationships
sns.pairplot(df[numerical_cols].dropna())
plt.show()

# 7. Handle Missing Values & Outliers
df_clean = df.dropna()
print("Shape after dropping missing values:", df_clean.shape)
plt.figure(figsize=(6,3))
sns.boxplot(x=df_clean['Experience'])
plt.title("Experience Boxplot")
plt.show()


# 8. Prepare for Modeling
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols)
print("Final encoded dataset shape:", df_encoded.shape)

