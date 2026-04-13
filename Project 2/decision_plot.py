import numpy as np
import matplotlib.pyplot as plt
from decision_tree import build_tree, predict  # import your functions
from decision_tree import Node  # Node class if needed

# --- Load Data ---
# Assuming label is last column
train_file = "heart_train.tsv"
test_file  = "heart_test.tsv"

# Load headers
with open(train_file) as f:
    header = f.readline().strip().split("\t")
features_idx = list(range(len(header) - 1))  # all columns except last

# Load data as strings (so tree works the same as before)
X_train = np.loadtxt(train_file, delimiter="\t", skiprows=1, usecols=features_idx, dtype=str)
y_train = np.loadtxt(train_file, delimiter="\t", skiprows=1, usecols=len(header)-1, dtype=str)

X_test  = np.loadtxt(test_file, delimiter="\t", skiprows=1, usecols=features_idx, dtype=str)
y_test  = np.loadtxt(test_file, delimiter="\t", skiprows=1, usecols=len(header)-1, dtype=str)

num_features = len(features_idx)

train_errors = []
test_errors = []

# --- Train trees for all depths ---
for depth in range(num_features + 1):
    # Build decision tree
    tree = build_tree(X_train, y_train, header, max_depth=depth)
    
    # Predict on training and testing data
    y_train_pred = predict(tree, X_train)
    y_test_pred = predict(tree, X_test)
    
    # Compute errors
    train_err = np.mean(y_train_pred != y_train.astype(int))
    test_err = np.mean(y_test_pred != y_test.astype(int))
    
    train_errors.append(train_err)
    test_errors.append(test_err)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(range(num_features + 1), train_errors, marker='o', label='Training Error')
plt.plot(range(num_features + 1), test_errors, marker='s', label='Testing Error')
plt.xlabel("Tree Depth")
plt.ylabel("Error Rate")
plt.title("Decision Tree Error vs Depth (Heart Dataset)")
plt.xticks(range(num_features + 1))
plt.grid(True)
plt.legend()
plt.show()

# Optional: save figure
plt.savefig("heart_error_vs_depth.png", dpi=300)