import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Styling ---
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

# --- Dataset ---
np.random.seed(42)
n_samples = 1000
n_features = 50

X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Experiment 1: MSE vs Leaf Nodes per Tree ---
leaf_nodes = np.linspace(2, 500, 20, dtype=int)
ensemble_sizes_fixed = [1, 5, 10, 50]
dt_test_errors = {ens: [] for ens in ensemble_sizes_fixed}

for leaves in leaf_nodes:
    for ens in ensemble_sizes_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=leaves, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        dt_test_errors[ens].append(mean_squared_error(y_test, rf.predict(X_test)))

# --- Experiment 2: MSE vs Ensemble Size ---
ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
tree_depths_fixed = [20, 50, 100, 500]
rf_test_errors = {depth: [] for depth in tree_depths_fixed}

for ens in ensemble_sizes:
    for depth in tree_depths_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_test_errors[depth].append(mean_squared_error(y_test, rf.predict(X_test)))

# --- Composite (Transition) Experiment ---
composite_leaf_nodes = np.linspace(2, 100, 25, dtype=int)
composite_ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
composite_test_errors = []

for leaves in composite_leaf_nodes:
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=42)
    tree.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, tree.predict(X_test)))

for ens in composite_ensemble_sizes:
    forest = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=1000, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)
    composite_test_errors.append(mean_squared_error(y_test, forest.predict(X_test)))

interpolation_idx = len(composite_leaf_nodes) - 1

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True, sharey=True)

# Panel A: Composite Double Descent Curve
composite_x_axis = [f"L{l}" for l in composite_leaf_nodes] + [f"E{e}" for e in composite_ensemble_sizes]
axes[0].plot(range(len(composite_x_axis)), composite_test_errors, color="red", label="Test Error")
axes[0].axvline(interpolation_idx, linestyle='dotted', color='black', linewidth=1.5, label="Transition Point")
axes[0].set_title("Double Descent Transition")
axes[0].set_xlabel("Model Complexity")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_xticks(range(0, len(composite_x_axis), max(len(composite_x_axis) // 6, 1)))
axes[0].set_xticklabels(
    [composite_x_axis[i] for i in range(0, len(composite_x_axis), max(len(composite_x_axis) // 6, 1))],
    rotation=20, ha="right")
axes[0].legend(frameon=True, facecolor='white')
axes[0].grid(False)

# Panel B: MSE vs Leaf Nodes (Fixed Trees)
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[1].plot(leaf_nodes, dt_test_errors[ens], marker='o', label=rf"$P_{{\mathrm{{ens}}}}$ = {ens}")
axes[1].set_title("Varying $P_{\\mathrm{leaf}}$ (Fixed Ensemble Size)")
axes[1].set_xlabel(r"$P_{\mathrm{leaf}}$")
axes[1].legend(frameon=True, facecolor='white')
axes[1].grid(False)

# Panel C: MSE vs Ensemble Size (Fixed Depth)
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^']
offset = np.linspace(-0.3, 0.3, len(tree_depths_fixed))
for i, depth in enumerate(tree_depths_fixed):
    jittered_x = [x + offset[i] for x in ensemble_sizes]
    axes[2].plot(jittered_x, rf_test_errors[depth],
                 label=rf"$P_{{\mathrm{{leaf}}}}$ = {depth}",
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 alpha=0.9)
axes[2].set_title("Varying $P_{\\mathrm{ens}}$ (Fixed Maximum Leaf Nodes)")
axes[2].set_xlabel(r"$P_{\mathrm{ens}}$")
axes[2].legend(frameon=True, facecolor='white')
axes[2].grid(False)

# Save and show
plt.savefig("styled_tree_experiments_left_composite.png", dpi=300)
plt.show()
