# =========================================================
# DECISION TREE (ID3) LAB ASSIGNMENT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import math

from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# =========================================================
# LOAD PLAY TENNIS DATASET
# =========================================================

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes',
                   'No', 'Yes', 'No', 'Yes', 'Yes',
                   'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("\n================ DATASET ================\n")
print(df)

# =========================================================
# ENTROPY FUNCTION
# =========================================================

def entropy(target_col):

    counts = np.bincount(target_col)

    probabilities = counts / len(target_col)

    entropy_value = 0

    for prob in probabilities:

        if prob > 0:
            entropy_value += -prob * math.log2(prob)

    return entropy_value

# =========================================================
# INFORMATION GAIN FUNCTION
# =========================================================

def information_gain(data, feature, target_name):

    total_entropy = entropy(data[target_name])

    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = 0

    for i in range(len(values)):

        subset = data.where(data[feature] == values[i]).dropna()

        subset_entropy = entropy(
            LabelEncoder().fit_transform(subset[target_name])
        )

        weighted_entropy += (counts[i] / np.sum(counts)) * subset_entropy

    info_gain = total_entropy - weighted_entropy

    return info_gain

# =========================================================
# LABEL ENCODING TARGET VARIABLE
# =========================================================

label_encoder = LabelEncoder()

df['PlayTennisEncoded'] = label_encoder.fit_transform(df['PlayTennis'])

# =========================================================
# ENTROPY OF DATASET
# =========================================================

print("\n================ ENTROPY OF DATASET ================\n")

dataset_entropy = entropy(df['PlayTennisEncoded'])

print("Entropy =", dataset_entropy)

# =========================================================
# INFORMATION GAIN FOR ALL ATTRIBUTES
# =========================================================

print("\n================ INFORMATION GAIN ================\n")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

for feature in features:

    ig = information_gain(df, feature, 'PlayTennisEncoded')

    print(f"{feature} --> Information Gain = {ig}")

# =========================================================
# ID3 ALGORITHM FROM SCRATCH
# =========================================================

def ID3(data, original_data, features, target_attribute_name):

    # If all target values are same
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If dataset is empty
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(
                np.unique(original_data[target_attribute_name],
                          return_counts=True)[1]
            )
        ]

    # If no features left
    elif len(features) == 0:
        return Counter(data[target_attribute_name]).most_common(1)[0][0]

    else:

        # Compute information gain
        item_values = [
            information_gain(data, feature, target_attribute_name)
            for feature in features
        ]

        # Best feature
        best_feature_index = np.argmax(item_values)

        best_feature = features[best_feature_index]

        # Create tree
        tree = {best_feature: {}}

        # Remove selected feature
        features = [i for i in features if i != best_feature]

        # Grow tree recursively
        for value in np.unique(data[best_feature]):

            subset = data.where(data[best_feature] == value).dropna()

            subtree = ID3(
                subset,
                original_data,
                features,
                target_attribute_name
            )

            tree[best_feature][value] = subtree

        return tree

# =========================================================
# BUILD DECISION TREE USING ID3
# =========================================================

print("\n================ ID3 DECISION TREE ================\n")

tree = ID3(
    df,
    df,
    features,
    'PlayTennisEncoded'
)

print(tree)

# =========================================================
# LIBRARY IMPLEMENTATION USING SCIKIT-LEARN
# =========================================================

print("\n================ SCIKIT-LEARN DECISION TREE ================\n")

# Encode categorical features
encoded_df = df.copy()

encoders = {}

for column in ['Outlook', 'Temperature', 'Humidity', 'Wind']:

    le = LabelEncoder()

    encoded_df[column] = le.fit_transform(encoded_df[column])

    encoders[column] = le

# Features and target
X = encoded_df[['Outlook', 'Temperature', 'Humidity', 'Wind']]

y = encoded_df['PlayTennisEncoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================================================
# MODEL TRAINING
# =========================================================

model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# VISUALIZE DECISION TREE
# =========================================================

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)

plt.title("Decision Tree using Scikit-Learn")

plt.show()

# =========================================================
# HYPERPARAMETER EXPERIMENTATION
# =========================================================

print("\n================ HYPERPARAMETER EXPERIMENTATION ================\n")

depth_values = [1, 2, 3, 4]

for depth in depth_values:

    temp_model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=depth,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    pred = temp_model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"Max Depth = {depth} --> Accuracy = {acc}")

# =========================================================
# OVERFITTING ANALYSIS
# =========================================================

print("\n================ OVERFITTING ANALYSIS ================\n")

train_accuracy = model.score(X_train, y_train)

test_accuracy = model.score(X_test, y_test)

print("Training Accuracy =", train_accuracy)

print("Testing Accuracy =", test_accuracy)

if train_accuracy > test_accuracy:
    print("\nPossible Overfitting Detected")

# =========================================================
# IMPROVEMENT USING DEPTH CONTROL
# =========================================================

print("\n================ IMPROVED MODEL ================\n")

pruned_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=2,
    min_samples_split=2,
    random_state=42
)

pruned_model.fit(X_train, y_train)

improved_pred = pruned_model.predict(X_test)

improved_accuracy = accuracy_score(y_test, improved_pred)

print("Improved Accuracy =", improved_accuracy)

# =========================================================
# FINAL RESULT
# =========================================================

print("\n================ RESULT ================\n")

print("Successfully implemented Decision Tree (ID3)")
print("using both scratch implementation and Scikit-Learn.")