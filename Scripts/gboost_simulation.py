import numpy as np
import matplotlib.pyplot as plt
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
ensemble_sizes = [1, 2, 5, 10, 20, 50, 80, 100]
ensemble_train_errors = []
ensemble_test_errors = []

for ens in ensemble_sizes:
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

    ensemble_train_errors.append(mean_squared_error(y_train, train_preds))
    ensemble_test_errors.append(mean_squared_error(y_test, test_preds))

# Experiment 3: Composite Complexity Axis (Boosting → Ensemble)
composite_train_mse = []
composite_test_mse = []
x_axis_composite = []

# Step 1: Varying Boosting Rounds (Fixed Ensemble Size)
for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    composite_train_mse.append(mean_squared_error(y_train, gb.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, gb.predict(X_test)))
    x_axis_composite.append(f"B{rounds}")

# Step 2: Varying Ensemble Size (Fixed Boosting Rounds)
for ens in ensemble_sizes:
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

    composite_train_mse.append(mean_squared_error(y_train, train_preds))
    composite_test_mse.append(mean_squared_error(y_test, test_preds))
    x_axis_composite.append(f"E{ens}")

# === Visualization: Multi-panel Figure with Single Y-axis Label and One Legend ===
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
axes[1].plot(ensemble_sizes, ensemble_train_errors, color='orange')
axes[1].plot(ensemble_sizes, ensemble_test_errors, color='green')
axes[1].set_xlabel("Number of Ensembled Models")
axes[1].set_title("Ensemble Size vs MSE")

# Panel 3: Composite Complexity Plot
axes[2].plot(range(len(x_axis_composite)), composite_train_mse, color="orange", linewidth=2)
axes[2].plot(range(len(x_axis_composite)), composite_test_mse, color="green", linewidth=2)

axes[2].set_xlabel("Model Complexity (Boosting → Ensemble)")
axes[2].set_title("Double Descent: Boosting → Ensemble")
axes[2].set_xticks(range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)))
axes[2].set_xticklabels(
    [x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))], rotation=20,
    ha="right", fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("double_descent_boosting_multipanel.png", dpi=300)
plt.show()
