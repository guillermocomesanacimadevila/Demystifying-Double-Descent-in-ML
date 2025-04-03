import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

# === Plotting Style === #
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# === Load and preprocess CRyPTIC data === #
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

# Load & label data
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]
susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0

resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]
resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1

# Combine and split
data = pd.concat([susceptible_df, resistant_df], ignore_index=True)
data.dropna(inplace=True)

X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === Params === #
P_boost_values = np.linspace(10, 300, 20, dtype=int)
P_ens_values = np.array([1, 2, 5, 10, 20])
fixed_boosting_rounds = [20, 50, 100, 200]
fixed_boost_rounds = 50

# === Composite Descent === #
composite_errors = []
composite_labels = []

for rounds in P_boost_values:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                   subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    composite_errors.append(mean_squared_error(y_test, y_pred))
    composite_labels.append(f"B{rounds}")

interp_idx = len(composite_errors) - 1

for ens in P_ens_values:
    preds = []
    for i in range(ens):
        gb = GradientBoostingRegressor(n_estimators=fixed_boost_rounds, max_depth=3, learning_rate=0.1,
                                       subsample=0.8, random_state=42 + i)
        gb.fit(X_train, y_train)
        preds.append(gb.predict(X_test))
    avg_preds = np.mean(preds, axis=0)
    composite_errors.append(mean_squared_error(y_test, avg_preds))
    composite_labels.append(f"E{ens}")

# === Vary Boosting Rounds (Fixed Ens Size) === #
boost_errors_by_ens = {}
for ens in P_ens_values:
    errs = []
    for rounds in P_boost_values:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                           subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        errs.append(mean_squared_error(y_test, avg_preds))
    boost_errors_by_ens[ens] = gaussian_filter1d(errs, sigma=1)

# === Vary Ensemble Size (Fixed Boosting Rounds) === #
ens_errors_by_boost = {}
for rounds in fixed_boosting_rounds:
    errs = []
    for ens in P_ens_values:
        preds = []
        for i in range(ens):
            gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                           subsample=0.8, random_state=42 + i)
            gb.fit(X_train, y_train)
            preds.append(gb.predict(X_test))
        avg_preds = np.mean(preds, axis=0)
        errs.append(mean_squared_error(y_test, avg_preds))
    ens_errors_by_boost[rounds] = gaussian_filter1d(errs, sigma=1)

# === Plotting === #
fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

# Panel A: Composite
axes[0].plot(range(len(composite_errors)), gaussian_filter1d(composite_errors, sigma=1), color='black')
axes[0].axvline(interp_idx, linestyle='--', color='gray', linewidth=2, label='Transition Point')
axes[0].set_xticks(range(len(composite_labels)))
axes[0].set_xticklabels(composite_labels, rotation=45)
axes[0].set_title("A. Double Descent in Gradient Boosting")
axes[0].set_xlabel("Model Complexity")
axes[0].set_ylabel("Mean Squared Error")
axes[0].legend()
axes[0].grid(False)

# Panel B: Boosting Rounds
colors = plt.cm.viridis(np.linspace(0, 1, len(P_ens_values)))
for i, ens in enumerate(P_ens_values):
    axes[1].plot(P_boost_values, boost_errors_by_ens[ens], label=fr"$P_{{ens}} = {ens}$", color=colors[i])
axes[1].set_title("B. Varying $P_{boost}$ (Fixed $P_{ens}$)")
axes[1].set_xlabel(r"$P_{boost}$")
axes[1].legend()
axes[1].grid(False)
axes[1].set_yticklabels([])
axes[1].set_ylabel("")

# Panel C: Ensemble Size
for i, rounds in enumerate(fixed_boosting_rounds):
    axes[2].plot(P_ens_values, ens_errors_by_boost[rounds], marker='o',
                 label=fr"$P_{{boost}} = {rounds}$", linestyle='-', alpha=0.85)
axes[2].set_title("C. Varying $P_{ens}$ (Fixed $P_{boost}$)")
axes[2].set_xlabel(r"$P_{ens}$")
axes[2].legend()
axes[2].grid(False)
axes[2].set_yticklabels([])
axes[2].set_ylabel("")

plt.savefig("gradientboosting_CRyPTIC_double_descent.png", dpi=300)
plt.show()