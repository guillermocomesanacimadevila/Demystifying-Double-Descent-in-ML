import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Paths to data directories
susceptible_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible/post_QC_susceptible"
resistant_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted/post_QC_resistant"

# Function to extract features from CSV files
def extract_features_from_csv(file_path):
    df = pd.read_csv(file_path)
    sample_col = df.columns[-1]
    df[["GT", "DP", "GT_CONF"]] = df[sample_col].str.split(":", expand=True)[[0, 1, 5]]
    
    # Convert genotype format to numerical values
    genotype_map = {"0/0": 0, "0/1": 1, "1/1": 2}
    df["GT"] = df["GT"].map(genotype_map).fillna(0).astype(int)  # Convert to numeric
    
    # Convert DP and GT_CONF to float
    df["DP"] = pd.to_numeric(df["DP"], errors="coerce").fillna(0)
    df["GT_CONF"] = pd.to_numeric(df["GT_CONF"], errors="coerce").fillna(0)
    
    return df[["POS", "GT", "DP", "GT_CONF"]]

# Load susceptible samples
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]
susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0  # Assign label 0 for susceptible

# Load resistant samples
resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]
resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1  # Assign label 1 for resistant

# Merge datasets
data = pd.concat([susceptible_df, resistant_df], ignore_index=True)
data.dropna(inplace=True)  # Drop NaNs (if any)

# Prepare training and test data
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Experiment 1: MSE vs Tree Depth (Decision Tree)
tree_depths = list(range(2, 51, 2))  # Varying depth for Decision Tree
dt_train_errors = []
dt_test_errors = []

for depth in tree_depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_errors.append(mean_squared_error(y_train, dt.predict(X_train)))
    dt_test_errors.append(mean_squared_error(y_test, dt.predict(X_test)))

# Experiment 2: MSE vs Ensemble Size (Random Forest)
fixed_depth = max(tree_depths)  # Fix tree depth at max depth
ensemble_sizes = [1, 2, 5, 10, 20, 50, 80, 100]
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

# === Visualization: Multi-panel Figure ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Panel 1: MSE vs Tree Depth (Decision Tree)
axes[0].plot(tree_depths, dt_train_errors, label="Train Error", color="blue")
axes[0].plot(tree_depths, dt_test_errors, label="Test Error", color="red")
axes[0].set_xlabel("Tree Depth")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title("Decision Tree: Depth vs MSE")
axes[0].legend()

# Panel 2: MSE vs Ensemble Size (Random Forest)
axes[1].plot(ensemble_sizes, rf_train_errors, label="Train Error", color="blue")
axes[1].plot(ensemble_sizes, rf_test_errors, label="Test Error", color="red")
axes[1].set_xlabel("Number of Trees in Ensemble")
axes[1].set_ylabel("Mean Squared Error")
axes[1].set_title("Random Forest: Trees vs MSE")
axes[1].legend()

plt.tight_layout()
plt.savefig("trees_real_data_multipanel.png", dpi=300)
plt.show()

# === Separate Composite Complexity Visualization ===
plt.figure(figsize=(7, 5), dpi=300)

plt.plot(range(len(x_axis_composite)), composite_train_mse, label="Train Error", color="blue", linewidth=2)
plt.plot(range(len(x_axis_composite)), composite_test_mse, label="Test Error", color="red", linewidth=2)

plt.xlabel("Model Complexity (Tree Depth → Ensemble)")
plt.ylabel("Mean Squared Error")
plt.title("Double Descent: Depth → Ensemble")

plt.xticks(range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)),
           [x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))],
           rotation=20, ha="right", fontsize=10)
plt.legend()

plt.savefig("trees_real_data_composite.png", dpi=300)
plt.show()