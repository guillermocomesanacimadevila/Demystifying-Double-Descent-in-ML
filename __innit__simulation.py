import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate biological data
np.random.seed(42)
n_samples = 1000
n_features = 50

# Generate random feature data
X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Tree-based models ===== #
# Experiment 1: MSE vs Tree Depth (Decision Tree)
tree_depths = np.arange(2, 51, 2)  
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

# ===== Gradient Boosting ===== #
# Experiment 1: MSE vs Boosting Rounds (Fixed Ensemble Size)
boosting_rounds = list(range(1, 301, 10))  # Reducing max boosting rounds for efficiency
gb_train_errors = []
gb_test_errors = []

for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    gb_train_errors.append(mean_squared_error(y_train, gb.predict(X_train)))
    gb_test_errors.append(mean_squared_error(y_test, gb.predict(X_test)))
    
# Experiment 2: MSE vs Ensemble Size (Fixed Boosting Rounds)
fixed_boost = 50
ensemble_sizes_gb = [1, 2, 5, 10, 20, 50, 80, 100]
ensemble_train_errors_gb = []
ensemble_test_errors_gb = []

for ens in ensemble_sizes_gb:
    models = [GradientBoostingRegressor(n_estimators=fixed_boost, max_depth=3, learning_rate=0.1, subsample=0.8,
                                        random_state=i) for i in range(ens)]

    train_preds = np.zeros_like(y_train, dtype=float)
    test_preds = np.zeros_like(y_test, dtype=float)

    for model in models:
        model.fit(X_train, y_train)
        train_preds += model.predict(X_train)
        test_preds += model.predict(X_test)

    train_preds /= ens
    test_preds /= ens

    ensemble_test_errors_gb.append(mean_squared_error(y_train, train_preds))
    ensemble_test_errors_gb.append(mean_squared_error(y_test, test_preds))
    
# Experiment 3: Composite Complexity Axis (Boosting → Ensemble)
composite_train_mse_gb = []
composite_test_mse_gb = []
x_axis_composite_gb = []

# Step 1: Varying Boosting Rounds (Fixed Ensemble Size)
for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    composite_train_mse.append(mean_squared_error(y_train, gb.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, gb.predict(X_test)))
    x_axis_composite.append(f"B{rounds}")

# Step 2: Varying Ensemble Size (Fixed Boosting Rounds)
for ens in ensemble_sizes_gb:
    models = [GradientBoostingRegressor(n_estimators=fixed_boost, max_depth=3, learning_rate=0.85, subsample=0.8,
                                        random_state=i) for i in range(ens)]

    train_preds = np.zeros_like(y_train, dtype=float)
    test_preds = np.zeros_like(y_test, dtype=float)

    for model in models:
        model.fit(X_train, y_train)
        train_preds += model.predict(X_train)
        test_preds += model.predict(X_test)

    train_preds /= ens
    test_preds /= ens

    composite_train_mse_gb.append(mean_squared_error(y_train, train_preds))
    composite_test_mse_gb.append(mean_squared_error(y_test, test_preds))
    x_axis_composite_gb.append(f"E{ens}")
    
# ===== Data Visualisation ===== #
fig, axes = plt.subplots(1, 3, figsize=(21, 5), dpi=300, sharey=True)  # Share y-axis

# Panel 1: MSE vs Tree Depth (Decision Tree)
axes[0].plot(tree_depths, dt_train_errors, color="blue", label="Train Error")
axes[0].plot(tree_depths, dt_test_errors, color="red", label="Test Error")
axes[0].set_xlabel("Tree Depth")
axes[0].set_ylabel("Mean Squared Error")  # Only appears in the first subplot
axes[0].set_title("Decision Tree: Depth vs MSE")

# Display legend only in the first subplot
axes[0].legend(loc="upper right")

# Panel 2: MSE vs Ensemble Size (Random Forest)
axes[1].plot(ensemble_sizes, rf_train_errors, color="blue")
axes[1].plot(ensemble_sizes, rf_test_errors, color="red")
axes[1].set_xlabel("Number of Trees in Ensemble")
axes[1].set_title("Random Forest: Trees vs MSE")

# Panel 3: Composite Complexity Axis (Tree Depth → Ensemble)
axes[2].plot(range(len(x_axis_composite)), composite_train_mse, color="blue", linewidth=2)
axes[2].plot(range(len(x_axis_composite)), composite_test_mse, color="red", linewidth=2)

axes[2].set_xlabel("Model Complexity (Tree Depth → Ensemble)")
axes[2].set_title("Double Descent: Depth → Ensemble")
axes[2].set_xticks(range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)))
axes[2].set_xticklabels([x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))], rotation=20, ha="right", fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("double_descent_trees_multipanel.png", dpi=300)
plt.show()

# Gradient Boosting
fig, axes = plt.subplots(1, 3, figsize=(21, 5), dpi=300, sharey=True)  # Share y-axis

# Panel 1: MSE vs Boosting Rounds
axes[0].plot(boosting_rounds, gb_train_errors, color='orange', label='Train Error')
axes[0].plot(boosting_rounds, gb_test_errors, color='green', label='Test Error')
axes[0].set_xlabel("Number of Boosting Rounds")
axes[0].set_ylabel("Mean Squared Error")  # Only in the first subplot
axes[0].set_title("Boosting Rounds vs MSE")

# Display legend only in the first subplot
axes[0].legend(loc="upper right")

# Panel 2: MSE vs Ensemble Size
axes[1].plot(ensemble_sizes_gb, ensemble_train_errors_gb, color='orange')
axes[1].plot(ensemble_sizes_gb, ensemble_train_errors_gb, color='green')
axes[1].set_xlabel("Number of Ensembled Models")
axes[1].set_title("Ensemble Size vs MSE")

# Panel 3: Composite Complexity Plot
axes[2].plot(range(len(x_axis_composite_gb)), composite_train_mse_gb, color="orange", linewidth=2)
axes[2].plot(range(len(x_axis_composite_gb)), composite_train_mse_gb, color="green", linewidth=2)

axes[2].set_xlabel("Model Complexity (Boosting → Ensemble)")
axes[2].set_title("Double Descent: Boosting → Ensemble")
axes[2].set_xticks(range(0, len(x_axis_composite_gb), max(len(x_axis_composite_gb) // 10, 1)))
axes[2].set_xticklabels(
    [x_axis_composite_gb[i] for i in range(0, len(x_axis_composite_gb), max(len(x_axis_composite_gb) // 10, 1))], rotation=20,
    ha="right", fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("double_descent_boosting_multipanel.png", dpi=300)
plt.show()