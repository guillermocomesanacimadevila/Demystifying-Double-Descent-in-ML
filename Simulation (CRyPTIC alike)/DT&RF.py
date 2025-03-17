import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set Seaborn theme for professional plots
sns.set_theme(style="whitegrid", context="talk")

# ========== Step 1: Generate Simulated SNP Data ==========

np.random.seed(42)
num_samples = 500  # Same as real experiment
num_snps = 10000   # Simulating genomic positions

# Simulate SNP presence/absence in binary format (0/1)
X_snp = np.random.randint(0, 2, size=(num_samples, num_snps))

# Simulate sequencing depth (DP) and genotype confidence (GT_CONF)
X_dp = np.random.normal(loc=30, scale=10, size=(num_samples, num_snps))  # Avg depth ~30
X_gt_conf = np.random.normal(loc=99, scale=5, size=(num_samples, num_snps))  # High confidence

# Simulate labels: 50% resistant (1), 50% susceptible (0)
y = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2))
y = y + np.random.normal(0, 0.1, size=y.shape)  # Small noise to simulate variability

# Shuffle dataset
idx = np.random.permutation(num_samples)
X_snp, X_dp, X_gt_conf, y = X_snp[idx], X_dp[idx], X_gt_conf[idx], y[idx]

# Convert into DataFrame format similar to real data
df_features = pd.DataFrame({
    "POS": np.tile(np.arange(1, num_snps + 1), num_samples),  # Simulated positions
    "GT": X_snp.flatten(),  # Flatten SNP presence/absence
    "DP": X_dp.flatten(),  # Flatten sequencing depth
    "GT_CONF": X_gt_conf.flatten()  # Flatten genotype confidence
})
df_labels = np.repeat(y, num_snps)  # Expand labels to match SNP count

# Combine into a structured dataset
df_simulated = df_features.copy()
df_simulated["label"] = df_labels  # Resistant (1) or Susceptible (0)

# ========== Step 2: Train/Test Split ==========
X = df_simulated.drop(columns=["label"])
y = df_simulated["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== Step 3: Define Model Complexity Ranges ==========
tree_depths = np.arange(2, 51, 2)
ensemble_sizes = np.arange(10, 201, 10)

# Store MSE values
dt_train_mse = []
dt_test_mse = []
rf_train_mse = []
rf_test_mse = []
composite_train_mse = []
composite_test_mse = []
x_axis_composite = []

# ========== Experiment 1: MSE vs Tree Depth (Decision Tree) ==========
for depth in tree_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    dt_train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    dt_test_mse.append(mean_squared_error(y_test, model.predict(X_test)))

# ========== Experiment 2: MSE vs Number of Trees (Random Forest) ==========
max_depth = max(tree_depths)
for ens in ensemble_sizes:
    model = RandomForestRegressor(n_estimators=ens, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    rf_train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    rf_test_mse.append(mean_squared_error(y_test, model.predict(X_test)))

# ========== Experiment 3: Composite Complexity Axis ==========
for depth in tree_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    composite_train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, model.predict(X_test)))
    x_axis_composite.append(f"D{depth}")  # Shorter label

for ens in ensemble_sizes:
    model = RandomForestRegressor(n_estimators=ens, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    composite_train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, model.predict(X_test)))
    x_axis_composite.append(f"T{ens}")  # Shorter label

# ========== Multipanel Figure: MSE vs Tree Depth & Ensemble Size ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Plot 1: MSE vs Tree Depth
axes[0].plot(tree_depths, dt_train_mse, label="Train Error", color="blue", linewidth=2)
axes[0].plot(tree_depths, dt_test_mse, label="Test Error", color="red", linewidth=2)
axes[0].set_xlabel("Tree Depth", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Mean Squared Error", fontsize=14, fontweight="bold")
axes[0].set_title("Test & Train Error vs Tree Depth (Single Tree)", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=12, frameon=True)

# Plot 2: MSE vs Number of Trees
axes[1].plot(ensemble_sizes, rf_train_mse, label="Train Error", color="blue", linewidth=2)
axes[1].plot(ensemble_sizes, rf_test_mse, label="Test Error", color="red", linewidth=2)
axes[1].set_xlabel("Number of Trees in Ensemble", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Mean Squared Error", fontsize=14, fontweight="bold")
axes[1].set_title("Test & Train Error vs Number of Trees (Ensemble)", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=12, frameon=True)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("multi_panel_mse_plot.png", dpi=300)
plt.show()

# ========== Separate Composite Complexity Plot ==========
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(range(len(x_axis_composite)), composite_train_mse, label="Train Error", color="blue", linewidth=2)
plt.plot(range(len(x_axis_composite)), composite_test_mse, label="Test Error", color="red", linewidth=2)
plt.xlabel("Model Complexity", fontsize=14, fontweight="bold")
plt.ylabel("Mean Squared Error", fontsize=14, fontweight="bold")
plt.title("Double Descent: Tree Depth → Ensemble Size", fontsize=16, fontweight="bold")

plt.xticks(
    range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)),
    [x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))],
    rotation=20, ha="right", fontsize=10
)
plt.legend(fontsize=12, frameon=True)
plt.savefig("composite_complexity_plot.png", dpi=300)
plt.show()
