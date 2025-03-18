import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate biological data (same dataset as Gradient Boosting)
np.random.seed(42)
n_samples = 1000
n_features = 50

# Generate random feature data
X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Experiment 1: MSE vs Tree Depth (Decision Tree)
tree_depths = np.arange(2, 51, 2)  # Varying depth for Decision Tree
dt_train_errors = []
dt_test_errors = []

for depth in tree_depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_errors.append(mean_squared_error(y_train, dt.predict(X_train)))
    dt_test_errors.append(mean_squared_error(y_test, dt.predict(X_test)))

# Experiment 2: MSE vs Ensemble Size (Random Forest)
fixed_depth = max(tree_depths)  # Fix tree depth at max depth
ensemble_sizes = [1, 2, 5, 10, 20, 50]
rf_train_errors = []
rf_test_errors = []

for ens in ensemble_sizes:
    rf = RandomForestRegressor(n_estimators=ens, max_depth=fixed_depth, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_errors.append(mean_squared_error(y_train, rf.predict(X_train)))
    rf_test_errors.append(mean_squared_error(y_test, rf.predict(X_test)))

# Experiment 3: Composite Complexity Axis (Tree Depth → Ensemble)
composite_train_mse = []
composite_test_mse = []
x_axis_composite = []

# Step 1: Varying Tree Depth (Fixed Single Tree)
for depth in tree_depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    composite_train_mse.append(mean_squared_error(y_train, dt.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, dt.predict(X_test)))
    x_axis_composite.append(f"D{depth}")  # Labeling for Decision Tree depth

# Step 2: Varying Ensemble Size (Fixed Tree Depth)
for ens in ensemble_sizes:
    rf = RandomForestRegressor(n_estimators=ens, max_depth=fixed_depth, random_state=42)
    rf.fit(X_train, y_train)
    composite_train_mse.append(mean_squared_error(y_train, rf.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, rf.predict(X_test)))
    x_axis_composite.append(f"E{ens}")  # Labeling for Ensemble Size

# === Visualization: Multi-panel Figure with Single Y-axis Label and One Legend ===
fig, axes = plt.subplots(1, 3, figsize=(21, 5), dpi=300, sharey=True)  # Share y-axis

# Panel 1: MSE vs Tree Depth (Decision Tree)
axes[0].plot(tree_depths, dt_train_errors, color="orange", label="Train Error")
axes[0].plot(tree_depths, dt_test_errors, color="green", label="Test Error")
axes[0].set_xlabel("Tree Depth")
axes[0].set_ylabel("Mean Squared Error")  # Only appears in the first subplot
axes[0].set_title("Decision Tree: Depth vs MSE")

# Display legend only in the first subplot
axes[0].legend(loc="upper right")

# Panel 2: MSE vs Ensemble Size (Random Forest)
axes[1].plot(ensemble_sizes, rf_train_errors, color="orange")
axes[1].plot(ensemble_sizes, rf_test_errors, color="green")
axes[1].set_xlabel("Number of Trees in Ensemble")
axes[1].set_title("Random Forest: Trees vs MSE")

# Panel 3: Composite Complexity Axis (Tree Depth → Ensemble)
axes[2].plot(range(len(x_axis_composite)), composite_train_mse, color="orange", linewidth=2)
axes[2].plot(range(len(x_axis_composite)), composite_test_mse, color="green", linewidth=2)

axes[2].set_xlabel("Model Complexity (Tree Depth → Ensemble)")
axes[2].set_title("Double Descent: Depth → Ensemble")
axes[2].set_xticks(range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)))
axes[2].set_xticklabels([x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))], rotation=20, ha="right", fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("double_descent_trees_multipanel.png", dpi=300)
plt.show()
