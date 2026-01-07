from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train baseline model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Baseline accuracy (no noise)
baseline_pred = model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("Baseline accuracy (no noise):", round(baseline_acc, 4))

# Test robustness under increasing noise
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print("\nAccuracy under noise:")
for noise in noise_levels:
    noise_matrix = np.random.normal(
        loc=0.0,
        scale=noise,
        size=X_test.shape
    )
    X_test_noisy = X_test + noise_matrix
    y_pred_noisy = model.predict(X_test_noisy)
    acc = accuracy_score(y_test, y_pred_noisy)
    print(f"Noise level {noise}: Accuracy = {round(acc, 4)}")
