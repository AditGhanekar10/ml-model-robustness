from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []

# Evaluate robustness
for noise in noise_levels:
    noise_matrix = np.random.normal(0, noise, X_test.shape)
    X_test_noisy = X_test + noise_matrix
    y_pred = model.predict(X_test_noisy)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot
plt.figure()
plt.plot(noise_levels, accuracies, marker='o')
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.title("Model Robustness: Accuracy vs Noise Level")
plt.grid(True)

# Save plot
plt.savefig("output/noise_robustness.png")
plt.show()

print("Plot saved as output/noise_robustness.png")
