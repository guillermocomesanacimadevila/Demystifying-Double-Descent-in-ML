import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

# Paths to data directories
susceptible_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_susceptible/gunziped_susceptible/csv_susceptible/post_QC_susceptible"
resistant_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted/post_QC_resistant"

# Function to extract features from CSV files
def extract_features_from_csv(file_path):
    df = pd.read_csv(file_path)
    sample_col = df.columns[-1]
    df[["GT", "DP", "GT_CONF"]] = df[sample_col].str.split(":", expand=True) [[0, 1, 5]]
    
    # Convert GT to numeric
    genotype_map = {"0/0": 0, "0/1": 1, "1/1": 2}
    df["GT"] = df["GT"].map(genotype_map).fillna(0).astype(int)
    
    # Convert DP and GT_CONF to float
    df["DP"] = pd.to_numeric(df["DP"], errors="coerce").fillna(0)
    df["GT_CONF"] = pd.to_numeric(df["GT_CONF"], errors="coerce").fillna(0)
    
    return df[["POS", "GT", "DP", "GT_CONF"]]

# Load susceptible samples
susceptible_data = [extract_features_from_csv(os.path.join(susceptible_dir, file))
                    for file in os.listdir(susceptible_dir) if file.endswith(".csv")]
susceptible_df = pd.concat(susceptible_data, ignore_index=True)
susceptible_df["label"] = 0  # Assign label 0 for susceptible

# Load resistant samples
resistant_data = [extract_features_from_csv(os.path.join(resistant_dir, file))
                  for file in os.listdir(resistant_dir) if file.endswith(".csv")]
resistant_df = pd.concat(resistant_data, ignore_index=True)
resistant_df["label"] = 1  # Assign label 1 for resistant

# Merge datasets
data = pd.concat([susceptible_df, resistant_df], ignore_index=True)
data.dropna(inplace=True)  # Drop NaNs (if any)

# Prepare training and test data
X = data.drop(columns=["label"])
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Experiment 1: MSE vs Boosting Rounds
boosting_rounds = list(range(10, 300, 20))
gb_train_errors = []
gb_test_errors = []

for n in boosting_rounds:
    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.85, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    
    gb_train_errors.append(mean_squared_error(y_train, gb.predict(X_train)))
    gb_test_errors.append(mean_squared_error(y_test, gb.predict(X_test)))

# Experiment 2: MSE vs Ensemble Size (Boosting as Ensemble)
fixed_boost = 50
ensemble_sizes = [1, 2, 5, 10, 20, 50]
ensemble_train_errors = []
ensemble_test_errors = []

for ens in ensemble_sizes:
    models = [GradientBoostingClassifier(n_estimators=fixed_boost, learning_rate=0.85, max_depth=3, random_state=i)
              for i in range(ens)]
    
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
for n in boosting_rounds:
    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.85, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    composite_train_mse.append(mean_squared_error(y_train, gb.predict(X_train)))
    composite_test_mse.append(mean_squared_error(y_test, gb.predict(X_test)))
    x_axis_composite.append(f"B{n}")

# Step 2: Varying Ensemble Size (Fixed Boosting Rounds)
for ens in ensemble_sizes:
    models = [GradientBoostingClassifier(n_estimators=fixed_boost, learning_rate=0.85, max_depth=3, random_state=i)
              for i in range(ens)]
    
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

# === Visualization: Multi-panel Figure ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Panel 1: MSE vs Boosting Rounds
axes[0].plot(boosting_rounds, gb_train_errors, label='Train Error', color='blue')
axes[0].plot(boosting_rounds, gb_test_errors, label='Test Error', color='red')
axes[0].set_xlabel("Number of Boosting Rounds")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title("Boosting Rounds vs MSE")
axes[0].legend()

# Panel 2: MSE vs Ensemble Size
axes[1].plot(ensemble_sizes, ensemble_train_errors, label='Train Error', color='blue')
axes[1].plot(ensemble_sizes, ensemble_test_errors, label='Test Error', color='red')
axes[1].set_xlabel("Number of Ensembled Models")
axes[1].set_ylabel("Mean Squared Error")
axes[1].set_title("Ensemble Size vs MSE")
axes[1].legend()

plt.tight_layout()
plt.savefig("boosting_real_data_multipanel.png", dpi=300)
plt.show()

# === Separate Composite Complexity Visualization ===
plt.figure(figsize=(7, 5), dpi=300)

plt.plot(range(len(x_axis_composite)), composite_train_mse, label="Train Error", color="blue", linewidth=2)
plt.plot(range(len(x_axis_composite)), composite_test_mse, label="Test Error", color="red", linewidth=2)

plt.xlabel("Model Complexity (Boosting → Ensemble)")
plt.ylabel("Mean Squared Error")
plt.title("Double Descent: Boosting → Ensemble")

plt.xticks(range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1)),
           [x_axis_composite[i] for i in range(0, len(x_axis_composite), max(len(x_axis_composite) // 10, 1))],
           rotation=20, ha="right", fontsize=10)
plt.legend()

plt.savefig("boosting_real_data_composite.png", dpi=300)
plt.show()