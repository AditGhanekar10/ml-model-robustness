from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Combine features and target
df = pd.concat([X, y], axis=1)

# Basic checks
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
