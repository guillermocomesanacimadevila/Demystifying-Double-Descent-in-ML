import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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

# Load and label samples
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]

susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0

resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]

resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1

# Combine data
data = pd.concat([susceptible_df, resistant_df], ignore_index=True)
data.dropna(inplace=True)

# Set X & y
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Experiment 1: MSE vs Boosting Rounds === #
boosting_rounds = np.linspace(10, 300, 20, dtype=int)
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

# === Experiment 2: MSE vs Ensemble Size === #
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

# === Composite Axis: Boosting Rounds → Ensembling === #
composite_test_errors = []
composite_x_labels = []

# Step 1: Vary boosting rounds
for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    composite_test_errors.append(mean_squared_error(y_test, y_pred))
    composite_x_labels.append(f"B{rounds}")

# Step 2: Vary ensemble size
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

# === Data Visualisation === #
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

# Panel 1: Error vs Boosting Rounds
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for i, ens in enumerate(ensemble_sizes_fixed):
    axes[0].plot(boosting_rounds, boosting_test_errors[ens], color=colors[i], linewidth=2, label=f"Trees={ens}")
axes[0].set_title("Error by Boosting Rounds (Fixed Trees)")
axes[0].set_xlabel("Boosting Rounds")
axes[0].set_ylabel("MSE")
axes[0].legend()

# Panel 2: Error vs Ensemble Size
colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, boost in enumerate(fixed_boosting_rounds):
    axes[1].plot(ensemble_sizes, ensemble_test_errors[boost], color=colors[i], linewidth=2, label=f"Boosting={boost}")
axes[1].set_title("Error by Ensemble Size (Fixed Boosting)")
axes[1].set_xlabel("Number of Trees")
axes[1].legend()
axes[1].set_yticklabels([])

plt.tight_layout()
plt.savefig("boosting_CRyPTIC_main.png", dpi=300)
plt.show()

# === Composite Double Descent Figure === #
plt.figure(figsize=(8, 5), dpi=300)

plt.plot(range(len(composite_test_errors)), composite_test_errors, color='black', linewidth=2, label="Test Error")
plt.axvline(interpolation_idx, linestyle='--', color='black', linewidth=2, label='Interpolation Threshold')
plt.title("Double Descent in Boosting")
plt.xlabel("Boosting Rounds → Ensembling")
plt.ylabel("Mean Squared Error")

plt.xticks(
    range(0, len(composite_x_labels), max(len(composite_x_labels) // 6, 1)),
    [composite_x_labels[i] for i in range(0, len(composite_x_labels), max(len(composite_x_labels) // 6, 1))],
    rotation=20, ha="right", fontsize=10
)
plt.legend()
plt.tight_layout()
plt.savefig("boosting_CRyPTIC_composite.png", dpi=300)
plt.show()
