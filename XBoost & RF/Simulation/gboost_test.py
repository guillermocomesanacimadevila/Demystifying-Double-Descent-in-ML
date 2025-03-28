import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
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

# --- Dataset from your VCF-style SNP sim ---
np.random.seed(42)
n_samples = 500
n_snps = 1100
n_noise_snps = 400

y = np.array([0] * 250 + [1] * 250)
np.random.shuffle(y)

def generate_vcf_features_with_noise(n_samples, n_snps, labels, n_noise_snps=400):
    X = []
    for i in range(n_samples):
        POS = np.sort(np.random.randint(1e3, 1e6, size=n_snps))
        SNP = np.random.randint(0, 3, size=n_snps)
        GT_CONF = np.random.normal(loc=20 + 10 * labels[i], scale=5, size=n_snps)
        PHQ = np.random.normal(loc=30, scale=10, size=n_snps)
        DP = np.random.poisson(lam=50, size=n_snps)
        informative = np.stack([SNP, POS / 1e6, PHQ, GT_CONF, DP], axis=1).flatten()
        noise_snps = np.random.randint(0, 3, size=n_noise_snps * 5)
        sample_flat = np.concatenate([informative, noise_snps])
        X.append(sample_flat)
    return np.array(X)

X = generate_vcf_features_with_noise(n_samples, n_snps, y, n_noise_snps)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

# Add label noise
y_train_noisy = y_train.copy()
flip_indices = np.random.choice(len(y_train_noisy), size=int(0.1 * len(y_train_noisy)), replace=False)
y_train_noisy[flip_indices] = 1 - y_train_noisy[flip_indices]

# --- Parameters ---
P_boost_values = np.linspace(1, 200, 20, dtype=int)
P_ens_values = np.array([1, 2, 5, 10, 20])
fixed_boosting_rounds = [10, 25, 50, 200]
fixed_boost_rounds = 200

# --- Experiment A: Composite double descent ---
composite_errors = []
composite_labels = []

for rounds in P_boost_values:
    gb = XGBRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85,
                      subsample=0.8, random_state=42,
                      tree_method='hist', predictor='auto', verbosity=0)
    gb.fit(X_train, y_train_noisy)
    y_pred = gb.predict(X_test)
    composite_errors.append(mean_squared_error(y_test, y_pred))
    composite_labels.append(f"B{rounds}")

interp_idx = len(composite_errors) - 1

for ens in P_ens_values:
    preds = []
    for i in range(ens):
        gb = XGBRegressor(n_estimators=fixed_boost_rounds, max_depth=3, learning_rate=0.85,
                          subsample=0.8, random_state=42 + i,
                          tree_method='hist', predictor='auto', verbosity=0)
        gb.fit(X_train, y_train_noisy)
        preds.append(gb.predict(X_test))
    avg_preds = np.mean(preds, axis=0)
    composite_errors.append(mean_squared_error(y_test, avg_preds))
    composite_labels.append(f"E{ens}")

# --- Experiment B: Varying P_boost (Fixed P_ens) ---
boost_errors_by_ens = {}
for ens in P_ens_values:
    errs = []
    for rounds in P_boost_values:
        preds = []
        for i in range(ens):
            gb = XGBRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85,
                              subsample=0.8, random_state=42 + i,
                              tree_method='hist', predictor='auto', verbosity=0)
            gb.fit(X_train, y_train_noisy)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        errs.append(mean_squared_error(y_test, avg_preds))
    boost_errors_by_ens[ens] = gaussian_filter1d(errs, sigma=1)

# --- Experiment C: Varying P_ens (Fixed P_boost) ---
ens_errors_by_boost = {}
for rounds in fixed_boosting_rounds:
    errs = []
    for ens in P_ens_values:
        preds = []
        for i in range(ens):
            gb = XGBRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.85,
                              subsample=0.8, random_state=42 + i,
                              tree_method='hist', predictor='auto', verbosity=0)
            gb.fit(X_train, y_train_noisy)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        errs.append(mean_squared_error(y_test, avg_preds))
    ens_errors_by_boost[rounds] = gaussian_filter1d(errs, sigma=1)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

# Panel A: Composite plot
axes[0].plot(range(len(composite_errors)), gaussian_filter1d(composite_errors, sigma=1), color='black')
axes[0].axvline(interp_idx, linestyle='--', color='gray', linewidth=2, label='Transition Point')
axes[0].set_xticks(range(len(composite_labels)))
axes[0].set_xticklabels(composite_labels, rotation=45)
axes[0].set_title("A. Double Descent in XGBoost")
axes[0].set_xlabel("Model Complexity")
axes[0].set_ylabel("Mean Squared Error")
axes[0].legend()
axes[0].grid(False)

# Panel B: Error vs P_boost
colors = plt.cm.viridis(np.linspace(0, 1, len(P_ens_values)))
for i, ens in enumerate(P_ens_values):
    axes[1].plot(P_boost_values, boost_errors_by_ens[ens], label=fr"$P_{{ens}} = {ens}$", color=colors[i])
axes[1].set_title("B. Varying $P_{boost}$ (Fixed $P_{ens}$)")
axes[1].set_xlabel(r"$P_{boost}$")
axes[1].legend()
axes[1].grid(False)
axes[1].set_yticklabels([])
axes[1].set_ylabel("")

# Panel C: Error vs P_ens
for i, rounds in enumerate(fixed_boosting_rounds):
    axes[2].plot(P_ens_values, ens_errors_by_boost[rounds], marker='o',
                 label=fr"$P_{{boost}} = {rounds}$", linestyle='-', alpha=0.85)
axes[2].set_title("C. Varying $P_{ens}$ (Fixed $P_{boost}$)")
axes[2].set_xlabel(r"$P_{ens}$")
axes[2].grid(False)
axes[2].legend()
axes[2].set_yticklabels([])
axes[2].set_ylabel("")

plt.show()
