# =========================================================
# MACHINE LEARNING LAB ASSIGNMENT
# Linear Regression using UCI Wine Quality Dataset
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# LOAD DATASET
# =========================================================

# Download the dataset from:
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# Keep the CSV file in the same folder as this Python file

df = pd.read_csv("winequality-red.csv", sep=';')

print("\n================ DATASET LOADED ================\n")
print(df.head())

# =========================================================
# DATA INSPECTION
# =========================================================

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ DATASET SHAPE ================\n")
print(df.shape)

print("\n================ NULL VALUES ================\n")
print(df.isnull().sum())

print("\n================ STATISTICAL SUMMARY ================\n")
print(df.describe())

# =========================================================
# DATA PREPROCESSING
# =========================================================

# Check duplicate rows
print("\n================ DUPLICATE ROWS ================\n")
print("Duplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)

# =========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------
# Distribution of Target Variable
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df['quality'], bins=10, kde=True)

plt.title("Distribution of Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Frequency")

plt.show()

# ---------------------------------------------------------
# Pairplot (optional but useful)
# ---------------------------------------------------------

# Uncomment if needed
# sns.pairplot(df)
# plt.show()

# =========================================================
# FEATURE SELECTION
# =========================================================

X = df.drop('quality', axis=1)
y = df['quality']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n================ TRAIN TEST SPLIT ================\n")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# =========================================================
# MODEL TRAINING
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n================ MODEL TRAINED ================\n")

# =========================================================
# MODEL COEFFICIENTS
# =========================================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)

print("\nIntercept:", model.intercept_)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

print("\n================ SAMPLE PREDICTIONS ================\n")

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(comparison.head(10))

# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n================ MODEL EVALUATION ================\n")

print("Mean Absolute Error (MAE):", mae)

print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R2 Score:", r2)

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.title("Actual vs Predicted Wine Quality")

plt.show()

# =========================================================
# RESIDUAL ERROR PLOT
# =========================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

sns.histplot(residuals, bins=30, kde=True)

plt.title("Residual Error Distribution")

plt.xlabel("Residual Error")

plt.show()

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================ ASSIGNMENT COMPLETED ================\n")
print("Linear Regression successfully implemented on")
print("UCI Wine Quality Dataset.")