import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Plot styling
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

# Experiment 1: Error vs Boosting Rounds (Fixed Ensemble Size = 1)
boosting_rounds = np.linspace(1, 300, 20, dtype=int)
ensemble_sizes_fixed = [1, 5, 10, 50]
boosting_test_errors = {ens: [] for ens in ensemble_sizes_fixed}

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

# Experiment 2: Error vs Ensemble Size (Fixed Boosting Rounds)
ensemble_sizes = np.linspace(1, 100, 15, dtype=int)
fixed_boosting_rounds = [20, 50, 100, 200]
ensemble_test_errors = {boost: [] for boost in fixed_boosting_rounds}

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

# Composite Axis: Boosting Rounds → Ensembling
composite_test_errors = []
composite_x_labels = []

for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    composite_test_errors.append(mean_squared_error(y_test, y_pred))
    composite_x_labels.append(f"B{rounds}")

fixed_boost_rounds = 50
for ens in ensemble_sizes:
    preds = []
    for i in range(ens):
        gb = GradientBoostingRegressor(n_estimators=fixed_boost_rounds, max_depth=3, learning_rate=0.1,
                                        subsample=0.8, random_state=42 + i)
        gb.fit(X_train, y_train)
        preds.append(gb.predict(X_test))
    avg_preds = np.mean(preds, axis=0)
    composite_test_errors.append(mean_squared_error(y_test, avg_preds))
    composite_x_labels.append(f"E{ens}")

interpolation_idx = np.argmax(composite_test_errors)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True)

# Panel 1: Error by Boosting Rounds
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(boosting_rounds, boosting_test_errors[ens], label=f"Trees={ens}", color=colors[i], linewidth=2)
axes[0].set_title("Error by Boosting Rounds (Fixed Trees)")
axes[0].set_xlabel("Boosting Rounds")
axes[0].set_ylabel("MSE")
axes[0].legend()

# Panel 2: Error by Ensemble Size
colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, boost in enumerate(fixed_boosting_rounds):
    axes[1].plot(ensemble_sizes, ensemble_test_errors[boost], label=f"Boosting={boost}", color=colors[i], linewidth=2)
axes[1].set_title("Error by Ensemble Size (Fixed Boosting)")
axes[1].set_xlabel("Number of Trees")
axes[1].legend()
axes[1].set_yticklabels([])

# Panel 3: Continuous Test Error Line
axes[2].plot(range(len(composite_test_errors)), composite_test_errors, color='black', linewidth=2, label="Test Error")
axes[2].axvline(interpolation_idx, linestyle='--', color='red', linewidth=2, label='Interpolation Threshold')
axes[2].set_title("Double Descent in Boosting")
axes[2].set_xlabel("Boosting Rounds → Ensembling")
axes[2].set_xticks(range(0, len(composite_x_labels), max(len(composite_x_labels)//10, 1)))
axes[2].set_xticklabels([composite_x_labels[i] for i in range(0, len(composite_x_labels), max(len(composite_x_labels)//10, 1))], rotation=20, ha="right")
axes[2].legend()
axes[2].set_yticklabels([])

plt.tight_layout()
plt.show()
