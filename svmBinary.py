# =========================================================
# SVM FOR BINARY CLASSIFICATION
# Heart Disease Dataset
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# LOAD DATASET
# =========================================================

# Replace with your CSV file name
df = pd.read_csv("heart.csv")

print("\n================ DATASET HEAD ================\n")
print(df.head())

# =========================================================
# DATASET UNDERSTANDING
# =========================================================

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ DATASET SHAPE ================\n")
print(df.shape)

print("\n================ NULL VALUES ================\n")
print(df.isnull().sum())

# =========================================================
# TARGET VARIABLE
# =========================================================

# Target column should be named 'target'
X = df.drop('target', axis=1)
y = df['target']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# LINEAR SVM MODEL
# =========================================================

linear_svm = SVC(kernel='linear')

linear_svm.fit(X_train, y_train)

linear_pred = linear_svm.predict(X_test)

# =========================================================
# RBF SVM MODEL
# =========================================================

rbf_svm = SVC(kernel='rbf')

rbf_svm.fit(X_train, y_train)

rbf_pred = rbf_svm.predict(X_test)

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(y_test, y_pred, model_name):

    print(f"\n================ {model_name} ================\n")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n")
    print(cm)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d')

    plt.title(f"Confusion Matrix - {model_name}")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# =========================================================
# MODEL EVALUATION
# =========================================================

evaluate_model(y_test, linear_pred, "Linear SVM")

evaluate_model(y_test, rbf_pred, "RBF SVM")

# =========================================================
# KERNEL COMPARISON
# =========================================================

print("\n================ KERNEL COMPARISON ================\n")

linear_acc = accuracy_score(y_test, linear_pred)
rbf_acc = accuracy_score(y_test, rbf_pred)

print("Linear Kernel Accuracy :", linear_acc)
print("RBF Kernel Accuracy    :", rbf_acc)

if linear_acc > rbf_acc:
    print("\nLinear kernel performed better.")
elif rbf_acc > linear_acc:
    print("\nRBF kernel performed better.")
else:
    print("\nBoth kernels performed similarly.")

# =========================================================
# FINAL RESULT
# =========================================================

print("\n================ RESULT ================\n")

print("Successfully implemented SVM for Binary Classification.")