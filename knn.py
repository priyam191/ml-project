# =========================================================
# LAB ASSIGNMENT
# KNN and K-Means Algorithms
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans

# =========================================================
# LOAD IRIS DATASET
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

# Convert into DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

df['target'] = y

print("\n================ IRIS DATASET ================\n")
print(df.head())

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# PART A : K-NEAREST NEIGHBORS (KNN)
# =========================================================

print("\n================ KNN CLASSIFICATION ================\n")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------
# TRAIN KNN MODEL
# ---------------------------------------------------------

k = 5

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

y_pred = knn.predict(X_test)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy for K = {k} :", accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================================
# ACCURACY FOR DIFFERENT K VALUES
# =========================================================

print("\n================ ACCURACY FOR DIFFERENT K VALUES ================\n")

k_values = range(1, 11)

accuracy_list = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    accuracy_list.append(acc)

    print(f"K = {k} --> Accuracy = {acc}")

# ---------------------------------------------------------
# PLOT K VALUE VS ACCURACY
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(k_values, accuracy_list, marker='o')

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")

plt.show()

# =========================================================
# PART B : K-MEANS CLUSTERING
# =========================================================

print("\n================ K-MEANS CLUSTERING ================\n")

# ---------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------

wcss = []

K_range = range(1, 11)

for i in K_range:

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# ---------------------------------------------------------
# DISPLAY WCSS VALUES
# ---------------------------------------------------------

print("WCSS Values:\n")

for i in range(len(wcss)):
    print(f"K = {i+1} --> WCSS = {wcss[i]}")

# ---------------------------------------------------------
# PLOT ELBOW GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(K_range, wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

# =========================================================
# APPLY K-MEANS
# =========================================================

optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X_scaled)

print("\nCluster Labels:\n")
print(y_kmeans)

# =========================================================
# VISUALIZATION OF CLUSTERS
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[y_kmeans == 0, 0],
    X_scaled[y_kmeans == 0, 1],
    label='Cluster 1'
)

plt.scatter(
    X_scaled[y_kmeans == 1, 0],
    X_scaled[y_kmeans == 1, 1],
    label='Cluster 2'
)

plt.scatter(
    X_scaled[y_kmeans == 2, 0],
    X_scaled[y_kmeans == 2, 1],
    label='Cluster 3'
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")

plt.legend()

plt.show()

# =========================================================
# OBSERVATIONS
# =========================================================

print("\n================ OBSERVATIONS ================\n")

print("1. KNN accuracy changes with different K values.")
print("2. K-Means uses WCSS to determine optimal K.")
print("3. Elbow point indicates the best number of clusters.")

# =========================================================
# VIVA QUESTIONS
# =========================================================

print("\n================ VIVA QUESTIONS ================\n")

print("1. Difference between KNN and K-Means?")
print("KNN is supervised learning.")
print("K-Means is unsupervised learning.\n")

print("2. What is Elbow Method?")
print("It is used to find optimal K using WCSS graph.\n")

print("3. Why scaling is important?")
print("Scaling ensures all features contribute equally.\n")

print("4. What are limitations of K-Means?")
print("Sensitive to outliers and requires predefined K.\n")

# =========================================================
# FINAL RESULT
# =========================================================

print("\n================ RESULT ================\n")

print("Successfully implemented KNN and K-Means algorithms.")