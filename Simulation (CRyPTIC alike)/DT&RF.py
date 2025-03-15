import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set Seaborn Style
sns.set_style("whitegrid")  # Use a clean background
sns.set_context("notebook")  # Optimal text sizes

# Step 1: Simulate Balanced Bacterial VCF-like SNP Data (Regression Setting)
np.random.seed(42)
num_samples = 500
num_snps = 10000

# Generate synthetic SNP matrix (binary presence/absence of mutations)
X = np.random.randint(0, 2, size=(num_samples, num_snps))
y = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2))
y = y + np.random.normal(0, 0.1, size=y.shape)  # Add slight noise

# Shuffle dataset
idx = np.random.permutation(num_samples)
X, y = X[idx], y[idx]

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=(y > 0.5), random_state=42)

# Define complexity ranges
tree_depths = np.arange(2, 100, 2)  # Increasing tree depth
ensemble_sizes = np.arange(1, 500, 20)  # Increasing ensemble size

# Store MSE values
dt_train_mse, dt_test_mse = [], []
rf_train_mse, rf_test_mse = [], []
composite_train_mse, composite_test_mse = [], []
x_axis_composite = []

# First Plot: MSE vs Pleaf (Single Decision Tree)
for depth in tree_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train[:, :1000], y_train)

    dt_train_mse.append(mean_squared_error(y_train, model.predict(X_train[:, :1000])))
    dt_test_mse.append(mean_squared_error(y_test, model.predict(X_test[:, :1000])))

# Second Plot: MSE vs Pens (Random Forest with Fixed Depth)
max_depth = max(tree_depths)
for ens in ensemble_sizes:
    model = RandomForestRegressor(n_estimators=ens, max_depth=max_depth, random_state=42)
    model.fit(X_train[:, :1000], y_train)

    rf_train_mse.append(mean_squared_error(y_train, model.predict(X_train[:, :1000])))
    rf_test_mse.append(mean_squared_error(y_test, model.predict(X_test[:, :1000])))

# Third Plot: Composite Axis (First Pleaf, then Pens)
for depth in tree_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train[:, :1000], y_train)

    composite_train_mse.append(mean_squared_error(y_train, model.predict(X_train[:, :1000])))
    composite_test_mse.append(mean_squared_error(y_test, model.predict(X_test[:, :1000])))
    x_axis_composite.append(f"Pleaf={depth}")

for ens in ensemble_sizes:
    model = RandomForestRegressor(n_estimators=ens, max_depth=max_depth, random_state=42)
    model.fit(X_train[:, :1000], y_train)

    composite_train_mse.append(mean_squared_error(y_train, model.predict(X_train[:, :1000])))
    composite_test_mse.append(mean_squared_error(y_test, model.predict(X_test[:, :1000])))
    x_axis_composite.append(f"Pens={ens}")

# Plot 1: MSE vs Pleaf
plt.figure(figsize=(10, 4), dpi=300)
sns.lineplot(x=tree_depths, y=dt_train_mse, label="Train Error", color="blue", linewidth=2)
sns.lineplot(x=tree_depths, y=dt_test_mse, label="Test Error", color="red", linewidth=2)
plt.xlabel("Tree Depth")
plt.ylabel("Mean Squared Error")
plt.title("Test & Train Error vs Tree Depth (Single Tree)")
plt.legend()
plt.show()

# Plot 2: MSE vs Pens
plt.figure(figsize=(10, 4), dpi=300)
sns.lineplot(x=ensemble_sizes, y=rf_train_mse, label="Train Error", color="blue", linewidth=2)
sns.lineplot(x=ensemble_sizes, y=rf_test_mse, label="Test Error", color="red", linewidth=2)
plt.xlabel("Number of Trees in Ensemble")
plt.ylabel("Mean Squared Error")
plt.title("Test & Train Error vs Number of Trees (Ensemble)")
plt.legend()
plt.show()

# Plot 3: Composite Axis (Pleaf → Pens)
plt.figure(figsize=(12, 5), dpi=300)
sns.lineplot(x=range(len(x_axis_composite)), y=composite_train_mse, label="Train Error", color="blue", linewidth=2)
sns.lineplot(x=range(len(x_axis_composite)), y=composite_test_mse, label="Test Error", color="red", linewidth=2)
plt.xlabel("Model Complexity (Pleaf → Pens)")
plt.ylabel("Mean Squared Error")
plt.title("Double Descent: Tree Depth to Ensemble Size")
plt.xticks(range(len(x_axis_composite)), x_axis_composite, rotation=45, ha="right", fontsize=10)
plt.legend()
plt.show()