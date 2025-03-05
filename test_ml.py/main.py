import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def parse_csv(location):
    with open(os.path.expanduser(location), "r") as file:
        return pd.read_csv(location, sep=",")

# Simulating GWAS dataset
np.random.seed(42)
num_snps = 1000

# Simulating LDL-c and CHD-risk Beta values (GWAS effect sizes)
beta_ldl = np.random.normal(0, 1, num_snps)  # LDL-c effect sizes
true_gamma = 0.5
beta_chd = true_gamma * beta_ldl + np.random.normal(0, 0.2, num_snps)  # CHD-risk effects with noise

# Create a dataset and save it
data = pd.DataFrame({"Beta_LDL_c": beta_ldl, "Beta_CHD_risk": beta_chd})
data.to_csv("GWAS_SNP_Effect_Simulation.csv", index=False)

# Step 2: Perform Regression by Increasing the Number of SNPs (Training & Test Errors)
snps_range = np.arange(50, num_snps - 50, 50)  # Ensuring test set has samples
train_errors, test_errors = [], []

for num_features in snps_range:
    train_size = int(0.8 * num_features)  # 80% of the selected SNPs for training
    X_train = beta_ldl[:train_size].reshape(-1, 1)
    y_train = beta_chd[:train_size]
    X_test = beta_ldl[train_size:num_features].reshape(-1, 1)
    y_test = beta_chd[train_size:num_features]

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on Training and Test Data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute Mean Squared Errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

# Step 3: Plot Training & Test Error vs. Number of SNPs (Publication-Quality)
plt.figure(figsize=(8, 5), dpi=300)
sns.set_style("ticks")

# Plot training and test errors
plt.plot(snps_range, train_errors, marker='o', linestyle='-', label="Training Error",
         color='blue', linewidth=2.5, markersize=7, alpha=0.8)
plt.plot(snps_range, test_errors, marker='s', linestyle='-', label="Test Error",
         color='orange', linewidth=2.5, markersize=7, alpha=0.8)

# Highlight interpolation threshold
plt.axvline(int(0.8 * num_snps), color='red', linestyle='--', linewidth=2, label="Interpolation Threshold")

# Labels and title with professional formatting
plt.xlabel("Number of SNPs Used in Regression", fontsize=14, fontweight='bold')
plt.ylabel("Mean Squared Error (MSE)", fontsize=14, fontweight='bold')
plt.title("Training & Test Error vs. Number of SNPs", fontsize=16, fontweight='bold')

# Improve legend with a border and spacing
plt.legend(fontsize=12, frameon=True, loc='upper right', edgecolor='black')

# Improve axis ticks and add minor ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(axis='both', which='major', length=6, width=1.5)
plt.tick_params(axis='both', which='minor', length=3, width=1)
plt.minorticks_on()

# Remove top and right spines for a cleaner look
sns.despine()

# Save as high-resolution publication-ready images
plt.savefig("Training_Test_Error_vs_SNPs.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig("Training_Test_Error_vs_SNPs.png", format="png", dpi=300, bbox_inches="tight")

# Show figure
plt.show()