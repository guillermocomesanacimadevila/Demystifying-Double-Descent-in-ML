import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set global plot style
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# === Load and preprocess CRyPTIC data === #

susceptible_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible/post_QC_susceptible"
resistant_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted/post_QC_resistant"

def extract_features_from_csv(file_path):
    df = pd.read_csv(file_path)
    sample_col = df.columns[-1]
    df[["GT", "DP", "GT_CONF"]] = df[sample_col].str.split(":", expand=True)[[0, 1, 5]]
    genotype_map = {"0/0": 0, "0/1": 1, "1/1": 2}
    df["GT"] = df["GT"].map(genotype_map).fillna(0).astype(int)
    df["DP"] = pd.to_numeric(df["DP"], errors="coerce").fillna(0)
    df["GT_CONF"] = pd.to_numeric(df["GT_CONF"], errors="coerce").fillna(0)
    return df[["POS", "GT", "DP", "GT_CONF"]]

# Process all CSVs within the susceptible and resistant directories
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]

susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0

resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]

resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1

# Split into X & y
data = pd.concat([susceptible_df, resistant_df], ignore_index=True).dropna()
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Experiment 1: MSE vs Leaf Nodes per Tree === #
leaf_nodes = np.linspace(2, 200, 15, dtype=int)
ensemble_sizes_fixed = [1, 5, 10, 50]
dt_test_errors = {ens: [] for ens in ensemble_sizes_fixed}

for leaves in leaf_nodes:
    for ens in ensemble_sizes_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=leaves, random_state=42)
        rf.fit(X_train, y_train)
        dt_test_errors[ens].append(mean_squared_error(y_test, rf.predict(X_test)))

# === Experiment 2: MSE vs Ensemble Size === #
ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
tree_depths_fixed = [10, 20, 50, 100]
rf_test_errors = {depth: [] for depth in tree_depths_fixed}

for ens in ensemble_sizes:
    for depth in tree_depths_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=depth, random_state=42)
        rf.fit(X_train, y_train)
        rf_test_errors[depth].append(mean_squared_error(y_test, rf.predict(X_test)))

# === Composite Transition: Depth → Ensembling === #
composite_leaf_nodes = np.linspace(2, 100, 20, dtype=int)
composite_ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
composite_test_errors = []

for leaves in composite_leaf_nodes:
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=42)
    tree.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, tree.predict(X_test)))

for ens in composite_ensemble_sizes:
    forest = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=200, random_state=42)
    forest.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, forest.predict(X_test)))

interpolation_idx = np.argmax(composite_test_errors)

# === Visualisation === #
# === Two-Panel Figure: Leaf Nodes & Ensemble Size === #
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

# Panel 1: Error vs Leaf Nodes
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(leaf_nodes, dt_test_errors[ens], color=colors[i], linewidth=2, label=f"Trees={ens}")
axes[0].set_title("Error by Leaf Nodes (Fixed Trees)")
axes[0].set_xlabel("Number of Leaf Nodes")
axes[0].set_ylabel("MSE")
axes[0].legend()

# Panel 2: Error vs Ensemble Size
colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, depth in enumerate(tree_depths_fixed):
    axes[1].plot(ensemble_sizes, rf_test_errors[depth], color=colors[i], linewidth=2, label=f"Depth={depth}")
axes[1].set_title("Error by Ensemble Size (Fixed Depth)")
axes[1].set_xlabel("Number of Trees")
axes[1].legend()
axes[1].set_yticklabels([])

plt.tight_layout()
plt.savefig("CRyPTIC_rf_df_main.png", dpi=300)
plt.show()

# === Composite Double Descent Figure === #
plt.figure(figsize=(8, 5), dpi=300)

composite_x_axis = [f"L{l}" for l in composite_leaf_nodes] + [f"E{e}" for e in composite_ensemble_sizes]
plt.plot(range(len(composite_x_axis)), composite_test_errors, color="black", linewidth=2, label="Test Error")
plt.axvline(interpolation_idx, color='black', linestyle='dashed', linewidth=2, label="Interpolation Threshold")

plt.title("Double Descent in Trees")
plt.xlabel("Depth → Ensembling")
plt.ylabel("MSE")
plt.xticks(
    range(0, len(composite_x_axis), max(len(composite_x_axis) // 6, 1)),
    [composite_x_axis[i] for i in range(0, len(composite_x_axis), max(len(composite_x_axis) // 6, 1))],
    rotation=20, ha="right", fontsize=10
)
plt.legend()
plt.tight_layout()
plt.savefig("CRyPTIC_rf_df_composite.png", dpi=300)
plt.show()
