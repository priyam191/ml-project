# =========================================================
# SVM FOR MULTI-CLASS CLASSIFICATION
# Iris Dataset
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# LOAD IRIS DATASET
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

df['target'] = y

print("\n================ IRIS DATASET ================\n")
print(df.head())

# =========================================================
# DATASET ANALYSIS
# =========================================================

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ CLASS LABELS ================\n")
print(np.unique(y))

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
# SVM USING ONE VS ONE (OvO)
# =========================================================

ovo_model = SVC(
    kernel='rbf',
    decision_function_shape='ovo',
    C=1,
    gamma='scale'
)

ovo_model.fit(X_train, y_train)

ovo_pred = ovo_model.predict(X_test)

# =========================================================
# SVM USING ONE VS REST (OvR)
# =========================================================

ovr_model = SVC(
    kernel='rbf',
    decision_function_shape='ovr',
    C=1,
    gamma='scale'
)

ovr_model.fit(X_train, y_train)

ovr_pred = ovr_model.predict(X_test)

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(y_test, y_pred, model_name):

    print(f"\n================ {model_name} ================\n")

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy :", accuracy)
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

evaluate_model(y_test, ovo_pred, "SVM OvO")

evaluate_model(y_test, ovr_pred, "SVM OvR")

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

print("\n================ HYPERPARAMETER TUNING ================\n")

C_values = [0.1, 1, 10]

gamma_values = [0.1, 1, 'scale']

for C in C_values:

    for gamma in gamma_values:

        model = SVC(
            kernel='rbf',
            C=C,
            gamma=gamma
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        print(f"C = {C}, Gamma = {gamma} --> Accuracy = {acc}")

# =========================================================
# COMPARISON OF OvO AND OvR
# =========================================================

print("\n================ OvO vs OvR ================\n")

ovo_acc = accuracy_score(y_test, ovo_pred)

ovr_acc = accuracy_score(y_test, ovr_pred)

print("OvO Accuracy :", ovo_acc)
print("OvR Accuracy :", ovr_acc)

if ovo_acc > ovr_acc:
    print("\nOvO performed better.")
elif ovr_acc > ovo_acc:
    print("\nOvR performed better.")
else:
    print("\nBoth models performed similarly.")

# =========================================================
# FINAL RESULT
# =========================================================

print("\n================ RESULT ================\n")

print("Successfully implemented SVM for Multi-Class Classification.")