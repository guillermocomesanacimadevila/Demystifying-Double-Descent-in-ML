import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

# --- Style config ---
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

# --- Simulated dataset ---
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.rand(n_samples, n_features)
y = np.sin(2 * np.pi * X[:, 0]) + np.log(X[:, 1] + 1) + 0.5 * np.random.randn(n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Experiment 1: Error vs P_boost (Fixed P_ens) ---
P_boost_values = np.linspace(1, 300, 20, dtype=int)
fixed_ensemble_sizes = [1, 5, 10, 50]
boosting_test_errors = {ens: [] for ens in fixed_ensemble_sizes}

for rounds in P_boost_values:
    for ens in fixed_ensemble_sizes:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                            subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        boosting_test_errors[ens].append(mean_squared_error(y_test, avg_preds))

# --- Experiment 2: Error vs P_ens (Fixed P_boost) ---
P_ens_values = np.linspace(1, 100, 15, dtype=int)
fixed_boosting_rounds = [20, 50, 100, 200]
ensemble_test_errors = {boost: [] for boost in fixed_boosting_rounds}

for ens in P_ens_values:
    for boost in fixed_boosting_rounds:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=boost, max_depth=3, learning_rate=0.1,
                                            subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        ensemble_test_errors[boost].append(mean_squared_error(y_test, avg_preds))

# --- Composite: P_boost to P_ens Double Descent Curve ---
composite_test_errors = []
composite_x_labels = []

for rounds in P_boost_values:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    composite_test_errors.append(mean_squared_error(y_test, y_pred))
    composite_x_labels.append(f"$P_{{boost}}={rounds}$")

fixed_boost_rounds = 50
for ens in P_ens_values:
    preds = []
    for i in range(ens):
        gb = GradientBoostingRegressor(n_estimators=fixed_boost_rounds, max_depth=3, learning_rate=0.1,
                                        subsample=0.8, random_state=42 + i)
        gb.fit(X_train, y_train)
        preds.append(gb.predict(X_test))
    avg_preds = np.mean(preds, axis=0)
    composite_test_errors.append(mean_squared_error(y_test, avg_preds))
    composite_x_labels.append(f"$P_{{ens}}={ens}$")

interpolation_idx = np.argmax(composite_test_errors)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharey=True, constrained_layout=True)

# Panel A: Composite Double Descent
axes[0].plot(range(len(composite_test_errors)), gaussian_filter1d(composite_test_errors, sigma=1), color='black', label="Test Error")
axes[0].axvline(interpolation_idx, linestyle='--', color='gray', linewidth=2, label='Interpolation Threshold')
axes[0].set_title("Double Descent in Boosting")
axes[0].set_xlabel(r"$P_{boost} \rightarrow P_{ens}$")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_xticks(range(0, len(composite_x_labels), max(len(composite_x_labels)//6, 1)))
axes[0].set_xticklabels([composite_x_labels[i] for i in range(0, len(composite_x_labels), max(len(composite_x_labels)//6, 1))], rotation=20, ha="right")
axes[0].legend(frameon=True, facecolor='white')
axes[0].grid(False)

# Panel B: Boosting Rounds (P_boost)
colors = plt.cm.tab10(np.linspace(0, 1, len(fixed_ensemble_sizes)))
for i, ens in enumerate(fixed_ensemble_sizes):
    errors_smoothed = gaussian_filter1d(boosting_test_errors[ens], sigma=1)
    axes[1].plot(P_boost_values, errors_smoothed, marker='o', label=fr"$P_{{ens}} = {ens}$", color=colors[i])
axes[1].set_title("Varying Boosting Rounds")
axes[1].set_xlabel(r"$P_{boost}$")
axes[1].legend(frameon=True, facecolor='white')
axes[1].grid(False)

# Panel C: Ensemble Size (P_ens)
for i, boost in enumerate(fixed_boosting_rounds):
    errors_smoothed = gaussian_filter1d(ensemble_test_errors[boost], sigma=1)
    axes[2].plot(P_ens_values, errors_smoothed, marker='s', label=fr"$P_{{boost}} = {boost}$", color=colors[i])
axes[2].set_title("Varying Ensemble Size")
axes[2].set_xlabel(r"$P_{ens}$")
axes[2].legend(frameon=True, facecolor='white')
axes[2].grid(False)

plt.show()
