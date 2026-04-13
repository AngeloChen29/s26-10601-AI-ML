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

def neg_log_likelihood(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the average negative log-likelihood (NLL) for logistic regression.

    Parameters:
        theta (np.ndarray): Parameter vector of shape (D,)
        X (np.ndarray): Feature matrix of shape (N, D)
        y (np.ndarray): Labels of shape (N,)

    Returns:
        float: The average negative log-likelihood.
    """
    z = X.dot(theta)
    p = sigmoid(z)

    # Clip to prevent log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll

def train(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_epoch: int,
    learning_rate: float,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
):
    N, D = X.shape
    train_nlls, val_nlls = [], []

    for epoch in range(num_epoch):
        for i in range(N):
            xi = X[i]
            yi = y[i]
            z = np.dot(theta, xi)
            p = sigmoid(z)
            grad = (p - yi) * xi
            theta -= learning_rate * grad

        # Record metrics after each epoch
        train_nlls.append(neg_log_likelihood(theta, X, y))
        if X_val is not None and y_val is not None:
            val_nlls.append(neg_log_likelihood(theta, X_val, y_val))

    return train_nlls, val_nlls

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

    # Set fixed hyperparameters for this experiment
    num_epoch = 1000
    learning_rate = 0.1

    theta = np.zeros(X_train.shape[1])
    train_nlls, val_nlls = train(theta, X_train, y_train, num_epoch, learning_rate, X_val, y_val)

    # === Plot NLL vs Epoch ===
    import matplotlib.pyplot as plt

    epochs = np.arange(1, num_epoch + 1)
    plt.figure()
    plt.plot(epochs, train_nlls, label="Train")
    plt.plot(epochs, val_nlls, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log-Likelihood")
    plt.title("Training and Validation NLL vs. Epoch (η = 0.1)")
    plt.legend()
    plt.savefig("nll_vs_epoch_eta0.1.png")
    plt.close()

    y_train_pred = predict(theta, X_train)
    y_test_pred = predict(theta, X_test)
    train_err = compute_error(y_train_pred, y_train)
    test_err = compute_error(y_test_pred, y_test)
    print(f"Train error: {train_err:.4f}")
    print(f"Test error:  {test_err:.4f}")

    learning_rates = [1e-1, 1e-2, 1e-3]
    num_epoch = 1000

    plt.figure()
    for eta in learning_rates:
        theta = np.zeros(X_train.shape[1])
        train_nlls, _ = train(theta, X_train, y_train, num_epoch, eta)
        plt.plot(range(1, num_epoch + 1), train_nlls, label=f"η = {eta}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log-Likelihood (Train)")
    plt.title("Training NLL vs Epoch for Different Learning Rates")
    plt.legend()
    plt.savefig("nll_vs_epoch_learningrates.png")
    plt.close()




    
