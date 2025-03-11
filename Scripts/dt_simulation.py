import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Simulate biological data
np.random.seed(42)
n_samples = 1000
n_features = 50

# Generate random feature data
X = np.random.rand(n_samples, n_features)

# Create a non-linear target variable with noise
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Experiment setup: Varying tree depth
tree_depths = list(range(1, 51))  # Complexity parameter for single trees

# Store errors for decision trees
tree_train_errors = []
tree_test_errors = []

# Decision Trees Experiment (Varying Depth)
for depth in tree_depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train_errors.append(mean_squared_error(y_train, y_train_pred))
    tree_test_errors.append(mean_squared_error(y_test, y_test_pred))

# Experiment setup: Varying number of trees in an ensemble
num_trees = list(range(1, 201, 10))  # Number of trees in the ensemble

# Store errors for ensembles
rf_train_errors = []
rf_test_errors = []

# Random Forest Experiment (Varying Number of Trees)
for n_trees in num_trees:
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    rf_train_errors.append(mean_squared_error(y_train, y_train_pred))
    rf_test_errors.append(mean_squared_error(y_test, y_test_pred))

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree Depth Error Plot
axs[0].plot(tree_depths, tree_train_errors, label='Train Error', color='blue')
axs[0].plot(tree_depths, tree_test_errors, label='Test Error', color='red')
axs[0].set_xlabel("Tree Depth")
axs[0].set_ylabel("Mean Squared Error")
axs[0].set_title("Test & Train Error vs Tree Depth (Single Tree)")
axs[0].legend()

# Random Forest Error Plot (Varying Number of Trees)
axs[1].plot(num_trees, rf_train_errors, label='Train Error', color='blue')
axs[1].plot(num_trees, rf_test_errors, label='Test Error', color='red')
axs[1].set_xlabel("Number of Trees in Ensemble")
axs[1].set_ylabel("Mean Squared Error")
axs[1].set_title("Test & Train Error vs Number of Trees (Ensemble)")
axs[1].legend()

plt.savefig("rf_dt.png", dpi=300)
plt.show()