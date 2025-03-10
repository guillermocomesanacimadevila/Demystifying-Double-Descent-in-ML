from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
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

# Adjusted Experiment Setup: Reduce number of boosting rounds for efficiency
boosting_rounds = list(range(1, 301, 10))  # Reduce max boosting rounds to 300

# Store errors for gradient boosting
gb_train_errors = []
gb_test_errors = []

# Gradient Boosting Experiment (Varying Number of Boosting Rounds)
for rounds in boosting_rounds:
    gb = GradientBoostingRegressor(n_estimators=rounds, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    y_train_pred = gb.predict(X_train)
    y_test_pred = gb.predict(X_test)
    gb_train_errors.append(mean_squared_error(y_train, y_train_pred))
    gb_test_errors.append(mean_squared_error(y_test, y_test_pred))

# Visualization
plt.figure(figsize=(7, 5))

# Gradient Boosting Error Plot
plt.plot(boosting_rounds, gb_train_errors, label='Train Error', color='blue')
plt.plot(boosting_rounds, gb_test_errors, label='Test Error', color='red')
plt.xlabel("Number of Boosting Rounds")
plt.ylabel("Mean Squared Error")
plt.title("Double Descent in Gradient Boosting (Optimized)")
plt.legend()

plt.savefig("grad_boost.png", dpi=300)
plt.show()