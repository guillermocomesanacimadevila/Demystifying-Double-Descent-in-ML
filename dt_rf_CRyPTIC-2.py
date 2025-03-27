import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict

# --- Plotting style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

# === Load and preprocess CRyPTIC data ===
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

# Load data
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]
susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0

resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]
resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1

# Combine and split
data = pd.concat([susceptible_df, resistant_df], ignore_index=True).dropna()
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Add label noise
y_train_noisy = y_train.copy()
n_noise = int(0.1 * len(y_train_noisy))
flip_indices = np.random.choice(len(y_train_noisy), size=n_noise, replace=False)
y_train_noisy.iloc[flip_indices] = 1 - y_train_noisy.iloc[flip_indices]

def evaluate(model):
    model.fit(X_train, y_train_noisy)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# --- Experiment 1: Double Descent Composite Curves ---
P_leaf_max_values = [50, 100, 200, 500]
P_ens_values = [1, 2, 5, 10, 20, 50]
all_possible_leaf_sizes = [2, 5, 10, 20, 50, 100, 200, 500]

all_curves = OrderedDict()
x_label_master = {}

for P_leaf_max in P_leaf_max_values:
    curve_errors = []
    curve_labels = []
    leaf_sizes = [l for l in all_possible_leaf_sizes if l <= P_leaf_max]

    for leaf in leaf_sizes:
        model = DecisionTreeRegressor(max_leaf_nodes=leaf, random_state=42)
        err = evaluate(model)
        curve_errors.append(err)
        curve_labels.append(f"L:{leaf}")

    for n_ens in P_ens_values:
        model = RandomForestRegressor(n_estimators=n_ens, max_leaf_nodes=P_leaf_max,
                                      bootstrap=False, random_state=42, n_jobs=-1)
        err = evaluate(model)
        curve_errors.append(err)
        curve_labels.append(f"RF:{n_ens}")

    smoothed = gaussian_filter1d(curve_errors, sigma=1)
    all_curves[P_leaf_max] = smoothed
    x_label_master[P_leaf_max] = curve_labels

# === Plot Composite Double Descent Curves ===
styles = {
    50: {'linestyle': '--', 'marker': 'o', 'color': 'tab:blue'},
    100: {'linestyle': '-.', 'marker': 's', 'color': 'tab:orange'},
    200: {'linestyle': ':', 'marker': '^', 'color': 'tab:green'},
    500: {'linestyle': '-', 'marker': 'D', 'color': 'tab:red'},
}

fig, ax = plt.subplots(figsize=(14, 7))
for P_leaf_max, smoothed in all_curves.items():
    labels = x_label_master[P_leaf_max]
    style = styles[P_leaf_max]
    interp_index = len([l for l in all_possible_leaf_sizes if l <= P_leaf_max]) - 1
    legend_label = rf"$P_{{\mathrm{{leaf}}}}$ = {P_leaf_max}"

    ax.plot(range(len(labels)), smoothed, label=legend_label,
            linestyle=style['linestyle'], marker=style['marker'],
            color=style['color'], linewidth=2, markersize=6)

    ax.axvline(x=interp_index, linestyle='dotted', color=style['color'], linewidth=1.5, alpha=0.7)

longest_label_set = max(x_label_master.values(), key=len)
ax.set_xticks(range(len(longest_label_set)))
ax.set_xticklabels(longest_label_set, rotation=45)
ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Model Complexity: Leaf Nodes → Ensemble Size")
ax.legend(title="Transition", title_fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig("CRyPTIC_combined_double_descent_curves.png", dpi=300)
plt.show()

# --- Experiment 2: Vary P_leaf for fixed ensemble sizes ---
P_leaf_values = [2, 5, 10, 20, 50, 100, 200, 300, 500]
fixed_ensemble_sizes = [1, 5, 10, 50]
depth_curves = {}

for p_ens in fixed_ensemble_sizes:
    errors = []
    for p_leaf in P_leaf_values:
        model = RandomForestRegressor(n_estimators=p_ens, max_leaf_nodes=p_leaf,
                                      bootstrap=False, random_state=42, n_jobs=-1)
        err = evaluate(model)
        errors.append(err)
    depth_curves[p_ens] = gaussian_filter1d(errors, sigma=1)

# --- Experiment 3: Vary P_ens for fixed max_leaf_nodes ---
fixed_tree_depths = [20, 50, 100, 500]
ensemble_curves = {}
P_ens_values = [1, 2, 5, 10, 20, 50]

for p_leaf in fixed_tree_depths:
    errors = []
    for p_ens in P_ens_values:
        model = RandomForestRegressor(n_estimators=p_ens, max_leaf_nodes=p_leaf,
                                      bootstrap=False, random_state=42, n_jobs=-1)
        err = evaluate(model)
        errors.append(err)
    ensemble_curves[p_leaf] = gaussian_filter1d(errors, sigma=1)

# === Plot both depth and ensemble size plots ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)

# Panel 1: Varying P_leaf
for p_ens, errors in depth_curves.items():
    axes[0].plot(P_leaf_values, errors, marker='o', label=rf"$P_{{\mathrm{{ens}}}}$ = {p_ens}")
axes[0].set_title("Varying Leaf Nodes (Fixed Ensemble Size)")
axes[0].set_xlabel(r"$P_{\mathrm{leaf}}$")
axes[0].set_ylabel("Mean Squared Error")
axes[0].legend(frameon=True, facecolor='white')
axes[0].grid(False)

# Panel 2: Varying P_ens
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^']
offset = np.linspace(-0.3, 0.3, len(fixed_tree_depths))

for i, (p_leaf, errors) in enumerate(ensemble_curves.items()):
    jittered_x = [x + offset[i] for x in P_ens_values]
    axes[1].plot(jittered_x, errors,
                 label=rf"$P_{{\mathrm{{leaf}}}}$ = {p_leaf}",
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 alpha=0.9)
axes[1].set_title("Varying Ensemble Size (Fixed Leaf Nodes)")
axes[1].set_xlabel(r"$P_{\mathrm{ens}}$")
axes[1].legend(frameon=True, facecolor='white')
axes[1].grid(False)

plt.savefig("CRyPTIC_curth_experiments_polished.png", dpi=300)
plt.show()
