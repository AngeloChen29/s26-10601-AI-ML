import sys
import math
import argparse
import numpy as np

class Node:
    def __init__(self, depth=0):
        self.left = None     
        self.right = None   
        self.attr = None     
        self.vote = None      
        self.depth = depth    # current depth
        self.zeros = 0       # number of label=0 at node
        self.ones = 0       # number of label=1 at node

def entropy(zeros, ones):
    tot = zeros + ones
    if tot == 0:
        return 0.0
    e = 0.0
    
    for c in (zeros, ones):
        if c != 0:
            p = c / tot
            e -= p * math.log2(p)
    return e

def mutual_information(X_col, y):
    total = len(y)
    zeros = np.sum(y == "0")
    ones = total - zeros
    hY = entropy(zeros, ones)

    mi = hY
    for v in [0, 1]:
        mask = (X_col == str(v))
        subset = y[mask]
        
        c0 = np.sum(subset == "0")
        c1 = len(subset) - c0
        h = entropy(c0, c1)

        mi -= (len(subset) / total) * h
    return mi

def majority_vote(y):
    zeros = np.sum(y == "0")
    ones = len(y) - zeros

    if zeros > ones:
        return 0
    else:
        return 1

def build_tree(X, y, header, max_depth, depth=0):
    node = Node(depth)
    node.zeros = np.sum(y == "0")
    node.ones = len(y) - node.zeros
    node.vote = majority_vote(y)

    if depth >= max_depth:
        return node

    best_attr = None
    best_mi = 0.0
    for i in range(X.shape[1]):
        mi = mutual_information(X[:, i], y)
        if mi > best_mi:
            best_mi = mi
            best_attr = i

    if best_attr is None:
        return node

    node.attr = best_attr

    X_left = []
    y_left = []
    X_right = []
    y_right = []

    for xi, yi in zip(X, y):
        if xi[best_attr] == '0':
            X_left.append(xi)
            y_left.append(yi)
        else:
            X_right.append(xi)
            y_right.append(yi)

    if X_left:
        node.left = build_tree(np.array(X_left), np.array(y_left), header, max_depth, depth + 1)
    if X_right:
        node.right = build_tree(np.array(X_right), np.array(y_right), header, max_depth, depth + 1)

    return node



def predict_one(node, x):
    while node.attr is not None:
        val = x[node.attr]
        if val == "0":
            if node.left is None:
                return node.vote
            node = node.left
        else:
            if node.right is None:
                return node.vote
            node = node.right
    return node.vote

def predict(node, X):
    return [predict_one(node, row) for row in X]

def error_rate(y_true, y_pred):
    mismatches = sum(int(yt != str(yp)) for yt, yp in zip(y_true, y_pred))
    return mismatches / len(y_true)

def print_tree(node, header, file, depth=0, branch=None):
    prefix = "| " * depth
    if branch is None:
        file.write(f"[{node.zeros} 0/{node.ones} 1]\n")
    else:
        file.write(f"{prefix}{header[branch[0]]} = {branch[1]}: [{node.zeros} 0/{node.ones} 1]\n")

    if node.attr is not None:
        if node.left is not None:
            print_tree(node.left, header, file, depth + 1, (node.attr, 0))
        if node.right is not None:
            print_tree(node.right, header, file, depth + 1, (node.attr, 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    train_data = np.genfromtxt(args.train_input, delimiter="\t", dtype=str, skip_header=1)
    test_data = np.genfromtxt(args.test_input, delimiter="\t", dtype=str, skip_header=1)

    with open(args.train_input) as f:
        header = f.readline().strip().split("\t")
    header = header[:-1]

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    tree = build_tree(X_train, y_train, header, args.max_depth)

    train_pred = predict(tree, X_train)
    test_pred = predict(tree, X_test)

    with open(args.train_out, "w") as f:
        for p in train_pred:
            f.write(f"{p}\n")
    with open(args.test_out, "w") as f:
        for p in test_pred:
            f.write(f"{p}\n")

    train_err = error_rate(y_train, train_pred)
    test_err = error_rate(y_test, test_pred)
    with open(args.metrics_out, "w") as f:
        f.write(f"error(train): {train_err:.6f}\n")
        f.write(f"error(test): {test_err:.6f}\n")

    with open(args.print_out, "w") as f:
        print_tree(tree, header, f)

if __name__ == "__main__":
    main()