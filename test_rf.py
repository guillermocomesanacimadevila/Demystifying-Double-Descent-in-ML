import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set global style for professional appearance
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Simulate dataset similar to the paper
np.random.seed(42)
n_samples = 1000
n_features = 50

X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Experiment 1: MSE vs Leaf Nodes per Tree for different ensemble sizes
leaf_nodes = np.linspace(2, 500, 20, dtype=int)
ensemble_sizes_fixed = [1, 5, 10, 50]

dt_test_errors = {ens: [] for ens in ensemble_sizes_fixed}

for leaves in leaf_nodes:
    for ens in ensemble_sizes_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=leaves, random_state=42)
        rf.fit(X_train, y_train)
        dt_test_errors[ens].append(mean_squared_error(y_test, rf.predict(X_test)))

# Experiment 2: MSE vs Ensemble Size for different tree depths
ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
tree_depths_fixed = [20, 50, 100, 500]

rf_test_errors = {depth: [] for depth in tree_depths_fixed}

for ens in ensemble_sizes:
    for depth in tree_depths_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=depth, random_state=42)
        rf.fit(X_train, y_train)
        rf_test_errors[depth].append(mean_squared_error(y_test, rf.predict(X_test)))

# Transition experiment: Connecting single trees to ensembles
composite_leaf_nodes = np.linspace(2, n_samples, 30, dtype=int)
composite_ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
composite_test_errors = []

# Stage 1: Increasing Pleaf (single trees, no ensembling)
for leaves in composite_leaf_nodes:
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=42)
    tree.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, tree.predict(X_test)))

# Stage 2: Transitioning to ensembles (Fixed max depth, increasing number of trees)
for ens in composite_ensemble_sizes:
    forest = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=n_samples, random_state=42)
    forest.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, forest.predict(X_test)))

# Determine interpolation threshold
interpolation_idx = np.argmax(composite_test_errors)  # Peak test error

# Create multipanel figure with professional styling
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1.2]})

# Panel 1: MSE vs Number of Leaf Nodes for different ensemble sizes
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(leaf_nodes, dt_test_errors[ens], color=colors[i], linestyle='-', linewidth=2, label=f"$P_{{ens}} = {ens}$")

axes[0].set_xlabel("$P_{leaf}$")
axes[0].set_ylabel("MSE")
axes[0].set_title("Error by $P_{leaf}$, $P_{ens}$ fixed")
axes[0].legend()

# Panel 2: MSE vs Ensemble Size for different tree depths
colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
for i, depth in enumerate(tree_depths_fixed):
    axes[1].plot(ensemble_sizes, rf_test_errors[depth], color=colors[i], linestyle='-', linewidth=2, label=f"$P_{{leaf}} = {depth}$")

axes[1].set_xlabel("$P_{ens}$")
axes[1].set_title("Error by $P_{ens}$, $P_{leaf}$ fixed")
axes[1].legend()

# Panel 3: Transition from single trees to ensembles
composite_x_axis = [f"L{l}" for l in composite_leaf_nodes] + [f"E{e}" for e in composite_ensemble_sizes]

# Split the curve into two colors
split_idx = len(composite_leaf_nodes)
axes[2].plot(range(split_idx), composite_test_errors[:split_idx], color="blue", linewidth=2, label="$P_{leaf}$ increasing")
axes[2].plot(range(split_idx, len(composite_x_axis)), composite_test_errors[split_idx:], color="red", linewidth=2, label="$P_{ens}$ increasing")
axes[2].axvline(interpolation_idx, color='black', linestyle='dashed', linewidth=2, label="Interpolation Threshold")

axes[2].set_xlabel("$P_{leaf} \times P_{ens}$")
axes[2].set_title("Double Descent in Trees")
axes[2].set_xticks(range(0, len(composite_x_axis), max(len(composite_x_axis) // 10, 1)))
axes[2].set_xticklabels([composite_x_axis[i] for i in range(0, len(composite_x_axis), max(len(composite_x_axis) // 10, 1))], rotation=20, ha="right", fontsize=10)
axes[2].legend()

plt.tight_layout()
plt.show()

