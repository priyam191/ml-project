# =========================================================
# LINEAR REGRESSION LAB ASSIGNMENT
# Parameter Estimation using:
# 1. Ordinary Least Squares (OLS)
# 2. Gradient Descent
# 3. Iterative Parameter Updates
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# LOAD DATASET
# =========================================================

# House Size (sq.ft)
X = np.array([800, 1000, 1200, 1500, 1800])

# House Price ($1000)
y = np.array([220, 270, 300, 350, 400])

# =========================================================
# DISPLAY DATASET
# =========================================================

data = pd.DataFrame({
    'House Size': X,
    'Price': y
})

print("\n================ DATASET ================\n")
print(data)

# =========================================================
# PART 1: ORDINARY LEAST SQUARES (OLS)
# =========================================================

print("\n================ OLS METHOD ================\n")

# Construct feature matrix
X_matrix = np.column_stack((np.ones(len(X)), X))

print("Feature Matrix X:\n")
print(X_matrix)

# Convert y into column vector
y_matrix = y.reshape(-1, 1)

print("\nOutput Vector y:\n")
print(y_matrix)

# OLS Formula
# Beta = (X^T X)^(-1) X^T y

XtX = X_matrix.T @ X_matrix

print("\nX^T X:\n")
print(XtX)

XtX_inverse = np.linalg.inv(XtX)

print("\n(X^T X)^-1:\n")
print(XtX_inverse)

Xty = X_matrix.T @ y_matrix

print("\nX^T y:\n")
print(Xty)

beta = XtX_inverse @ Xty

print("\nParameter Vector Beta:\n")
print(beta)

beta_0 = beta[0][0]
beta_1 = beta[1][0]

print("\nIntercept (β0):", beta_0)
print("Slope (β1):", beta_1)

# Final Equation
print("\nFinal Regression Equation:")
print(f"y = {beta_0:.2f} + {beta_1:.4f}x")

# =========================================================
# PART 2: PREDICTIONS
# =========================================================

y_pred_ols = beta_0 + beta_1 * X

print("\nPredicted Values:\n")
print(y_pred_ols)

# =========================================================
# PART 3: COST FUNCTION (MSE)
# =========================================================

mse = np.mean((y - y_pred_ols) ** 2)

print("\n================ COST FUNCTION ================\n")
print("Mean Squared Error (MSE):", mse)

# =========================================================
# PART 4: GRADIENT DESCENT
# =========================================================

print("\n================ GRADIENT DESCENT ================\n")

# Initialize parameters
b0 = 0
b1 = 0

# Learning rate
alpha = 0.0000001

# Number of iterations
iterations = 1000

# Number of samples
n = len(X)

# Store cost values
cost_history = []

for i in range(iterations):

    # Predicted values
    y_pred = b0 + b1 * X

    # Errors
    error = y - y_pred

    # Gradients
    db0 = (-2 / n) * np.sum(error)
    db1 = (-2 / n) * np.sum(X * error)

    # Update parameters
    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1

    # Compute cost
    cost = np.mean(error ** 2)

    # Store cost
    cost_history.append(cost)

# Final parameters
print("Final Intercept (β0):", b0)
print("Final Slope (β1):", b1)

print("\nRegression Equation from Gradient Descent:")
print(f"y = {b0:.2f} + {b1:.4f}x")

# =========================================================
# PART 5: ITERATIVE PARAMETER ESTIMATION
# =========================================================

print("\n================ ITERATIVE ESTIMATION ================\n")

b0_iter = 0
b1_iter = 0

for i in range(5):

    y_pred_iter = b0_iter + b1_iter * X

    error_iter = y - y_pred_iter

    db0_iter = (-2 / n) * np.sum(error_iter)
    db1_iter = (-2 / n) * np.sum(X * error_iter)

    b0_iter = b0_iter - alpha * db0_iter
    b1_iter = b1_iter - alpha * db1_iter

    cost_iter = np.mean(error_iter ** 2)

    print(f"Iteration {i+1}")
    print(f"β0 = {b0_iter:.6f}")
    print(f"β1 = {b1_iter:.6f}")
    print(f"Cost = {cost_iter:.6f}\n")

# =========================================================
# PART 6: VISUALIZATION
# =========================================================

# ---------------------------------------------------------
# Scatter Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(X, y)

plt.xlabel("House Size (sq.ft)")
plt.ylabel("Price ($1000)")
plt.title("Scatter Plot of Dataset")

plt.show()

# ---------------------------------------------------------
# Regression Line
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(X, y, label="Actual Data")

plt.plot(X, y_pred_ols, label="Regression Line")

plt.xlabel("House Size (sq.ft)")
plt.ylabel("Price ($1000)")
plt.title("Linear Regression Line")

plt.legend()

plt.show()

# ---------------------------------------------------------
# Cost Function Convergence
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(range(iterations), cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")

plt.show()

# =========================================================
# PART 7: THEORY QUESTIONS
# =========================================================

print("\n================ THEORY QUESTIONS ================\n")

print("1. Difference between OLS and Gradient Descent:")
print("OLS gives direct analytical solution.")
print("Gradient Descent finds parameters iteratively.\n")

print("2. Why learning rate is important?")
print("Learning rate controls the step size during parameter updates.\n")

print("3. What happens if learning rate is too large?")
print("The model may overshoot and fail to converge.\n")

print("4. Why does Gradient Descent require iterations?")
print("Because parameters are updated gradually to minimize cost.\n")

print("5. When is Gradient Descent preferred?")
print("For very large datasets where OLS becomes computationally expensive.\n")

# =========================================================
# END OF PROGRAM
# =========================================================

print("\n================ PROGRAM COMPLETED ================\n")