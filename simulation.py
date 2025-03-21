import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set global style for professional appearance
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Simulated dataset
np.random.seed(42)
n_samples = 1000
n_features = 50

X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Random Forest & Decision Tree Experiments ---
leaf_nodes = np.linspace(2, 500, 20, dtype=int)
ensemble_sizes_fixed = [1, 5, 10, 50]
dt_test_errors = {ens: [] for ens in ensemble_sizes_fixed}
ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
tree_depths_fixed = [20, 50, 100, 500]
rf_test_errors = {depth: [] for depth in tree_depths_fixed}
rf_composite_test_errors = []
rf_composite_x_labels = []

# MSE vs Leaf Nodes
for leaves in leaf_nodes:
    for ens in ensemble_sizes_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=leaves, random_state=42)
        rf.fit(X_train, y_train)
        dt_test_errors[ens].append(mean_squared_error(y_test, rf.predict(X_test)))

# MSE vs Ensemble Size
for ens in ensemble_sizes:
    for depth in tree_depths_fixed:
        rf = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=depth, random_state=42)
        rf.fit(X_train, y_train)
        rf_test_errors[depth].append(mean_squared_error(y_test, rf.predict(X_test)))

# Composite Experiment
for leaves in leaf_nodes:
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=42)
    tree.fit(X_train, y_train)
    rf_composite_test_errors.append(mean_squared_error(y_test, tree.predict(X_test)))
    rf_composite_x_labels.append(f"L{leaves}")

for ens in ensemble_sizes:
    forest = RandomForestRegressor(n_estimators=ens, max_leaf_nodes=500, random_state=42)
    forest.fit(X_train, y_train)
    rf_composite_test_errors.append(mean_squared_error(y_test, forest.predict(X_test)))
    rf_composite_x_labels.append(f"E{ens}")

# Plot Random Forest & Decision Tree Experiments
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.2]})
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(leaf_nodes, dt_test_errors[ens], color=colors[i], linewidth=2, label=f"Trees={ens}")
axes[0].set_title("Error by Leaf Nodes (Fixed Trees)")
axes[0].set_xlabel("Number of Leaf Nodes")
axes[0].set_ylabel("MSE")
axes[0].legend()

colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, depth in enumerate(tree_depths_fixed):
    axes[1].plot(ensemble_sizes, rf_test_errors[depth], color=colors[i], linewidth=2, label=f"Depth={depth}")
axes[1].set_title("Error by Ensemble Size (Fixed Depth)")
axes[1].set_xlabel("Number of Trees")
axes[1].legend()
axes[1].set_yticklabels([])

axes[2].plot(range(len(rf_composite_x_labels)), rf_composite_test_errors, color='black', linewidth=2)
axes[2].set_title("Double Descent in Trees")
axes[2].set_xlabel("Leaf Nodes → Ensembling")
axes[2].set_xticks(range(0, len(rf_composite_x_labels), max(len(rf_composite_x_labels) // 6, 1)))
axes[2].set_xticklabels([rf_composite_x_labels[i] for i in range(0, len(rf_composite_x_labels), max(len(rf_composite_x_labels) // 6, 1))], rotation=20, ha="right", fontsize=10)
axes[2].legend()
axes[2].set_yticklabels([])

plt.tight_layout()
plt.show()

# --- Gradient Boosting Experiments ---
boosting_rounds = np.linspace(1, 300, 20, dtype=int)
boosting_test_errors = {ens: [] for ens in ensemble_sizes_fixed}
fixed_boosting_rounds = [20, 50, 100, 200]
ensemble_test_errors = {boost: [] for boost in fixed_boosting_rounds}
gb_composite_test_errors = []
gb_composite_x_labels = []

# MSE vs Boosting Rounds
for rounds in boosting_rounds:
    for ens in ensemble_sizes_fixed:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                            subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        boosting_test_errors[ens].append(mean_squared_error(y_test, avg_preds))

# MSE vs Ensemble Size
for ens in ensemble_sizes:
    for boost in fixed_boosting_rounds:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=boost, max_depth=3, learning_rate=0.1,
                                            subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        ensemble_test_errors[boost].append(mean_squared_error(y_test, avg_preds))

# Composite Experiment
for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    gb_composite_test_errors.append(mean_squared_error(y_test, y_pred))
    gb_composite_x_labels.append(f"B{rounds}")

for ens in ensemble_sizes:
    preds = []
    for i in range(ens):
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                        subsample=0.8, random_state=42 + i)
        gb.fit(X_train, y_train)
        preds.append(gb.predict(X_test))
    avg_preds = np.mean(preds, axis=0)
    gb_composite_test_errors.append(mean_squared_error(y_test, avg_preds))
    gb_composite_x_labels.append(f"E{ens}")

# Plot Gradient Boosting Experiments
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.2]})

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(boosting_rounds, boosting_test_errors[ens], color=colors[i], linewidth=2, label=f"Trees={ens}")
axes[0].set_title("Error by Boosting Rounds (Fixed Trees)")
axes[0].set_xlabel("Boosting Rounds")
axes[0].set_ylabel("MSE")
axes[0].legend()

colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, boost in enumerate(fixed_boosting_rounds):
    axes[1].plot(ensemble_sizes, ensemble_test_errors[boost], color=colors[i], linewidth=2, label=f"Boosting={boost}")
axes[1].set_title("Error by Ensemble Size (Fixed Boosting)")
axes[1].set_xlabel("Number of Trees")
axes[1].legend()
axes[1].set_yticklabels([])

axes[2].plot(range(len(gb_composite_x_labels)), gb_composite_test_errors, color='black', linewidth=2)
axes[2].set_title("Double Descent in Boosting")
axes[2].set_xlabel("Boosting Rounds → Ensembling")
axes[2].set_xticks(range(0, len(gb_composite_x_labels), max(len(gb_composite_x_labels) // 6, 1)))
axes[2].set_xticklabels([gb_composite_x_labels[i] for i in range(0, len(gb_composite_x_labels), max(len(gb_composite_x_labels) // 6, 1))], rotation=20, ha="right", fontsize=10)
axes[2].legend()
axes[2].set_yticklabels([])

plt.tight_layout()
plt.show()
