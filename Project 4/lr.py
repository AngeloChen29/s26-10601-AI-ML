import numpy as np
import argparse

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    N, D = X.shape
    for epoch in range(num_epoch):
        for i in range(N):
            xi = X[i]
            yi = y[i]

            z = np.dot(theta, xi)
            p = sigmoid(z)

            grad = (p - yi) * xi
            theta -= learning_rate * grad

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    prob = sigmoid(X.dot(theta))
    return (prob >= 0.5).astype(int)


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    return float(np.mean(y_pred != y))

def add_bias(X):
    N = X.shape[0]
    return np.hstack([np.ones((N,1)), X])

def load_formatted(file):
    labels = []
    feats = []

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            parts = line.split('\t')
            labels.append(int(float(parts[0])))
            feats.append([float(x) for x in parts[1:]])
            

    X = np.array(feats)
    y = np.array(labels)

    return X, y

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    X_train, y_train = load_formatted(args.train_input)
    X_val, y_val = load_formatted(args.validation_input)
    X_test, y_test = load_formatted(args.test_input)

    X_train = add_bias(X_train)
    X_val = add_bias(X_val)
    X_test = add_bias(X_test)

    D = X_train.shape[1]
    theta = np.zeros(D)

    train(theta, X_train, y_train, args.num_epoch, args.learning_rate)

    y_train_pred = predict(theta, X_train)
    y_test_pred  = predict(theta, X_test)

    train_err = compute_error(y_train_pred, y_train)
    test_err  = compute_error(y_test_pred, y_test)

    with open(args.train_out, 'w') as out:
        for x in y_train_pred:
            out.write(f"{int(x)}\n")

    with open(args.test_out, 'w') as out:
        for x in y_test_pred:
            out.write(f"{int(x)}\n")

    with open(args.metrics_out, 'w') as f:
        f.write(f"error(train): {train_err:.6f}\n")
        f.write(f"error(test): {test_err:.6f}\n")




    
