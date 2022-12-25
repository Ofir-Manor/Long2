from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftSVM(BaseEstimator, ClassifierMixin):
    """
    Custom C-Support Vector Classification.
    """

    def __init__(self, C: float, lr: float = 1e-5, batch_size=32):
        """
        Initialize an instance of this class.
        ** Do not edit this method **

        :param C: inverse strength of regularization. Must be strictly positive.
        :param lr: the SGD learning rate (step size)
        """
        self.C = C
        self.lr = lr
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0

    # Initialize a random weight vector
    def init_solution(self, n_features: int):
        """
        Randomize an initial solution (weight vector)
        ** Do not edit this method **

        :param n_features:
        """
        self.w = np.random.randn(n_features)
        self.b = 0.0

    @staticmethod
    def loss(w: np.ndarray, b: float, C: float, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the SVM objective loss.

        :param w: weight vector for linear classification; array of shape (n_features,)
        :param b: bias scalar for linear classification
        :param C: inverse strength of regularization. Must be strictly positive.
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: the Soft SVM objective loss (float scalar)
        """
        margins = (X.dot(w) + b).reshape(-1, 1)
        hinge_inputs = np.multiply(margins, y.reshape(-1, 1))
        # Calculate the loss:
        hinge_loss = np.maximum(0, 1 - hinge_inputs)
        return np.power(np.linalg.norm(w), 2) + C * np.sum(hinge_loss)

    @staticmethod
    def __func_z(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Auxiliary private method to calculate f(z).
        """
        margins = (X.dot(w) + b).reshape(-1, 1)
        z = np.multiply(margins, y.reshape(-1, 1))
        return np.where(z < 1, -1, 0)

    @staticmethod
    def subgradient(w: np.ndarray, b: float, C: float, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Compute the (analytical) SVM objective sub-gradient.

        :param w: weight vector for linear classification; array of shape (n_features,)
        :param b: bias scalar for linear classification
        :param C: inverse strength of regularization. Must be strictly positive.
        :param X: samples for loss computation; array of shape (n_samples, n_features)
        :param y: targets for loss computation; array of shape (n_samples,)
        :return: a tuple with (the gradient of the weights, the gradient of the bias)
        """
        # Pre-calculations:
        func_z_vector = SoftSVM.__func_z(w, b, X, y)
        func_z_y = np.multiply(func_z_vector, y.reshape(-1, 1))
        # The analytical sub-gradient of soft-SVM w.r.t w:
        g_w = np.multiply(2, w) + np.multiply(C, X.T.dot(func_z_y).reshape(-1))
        # The analytical sub-gradient of soft-SVM w.r.t b:
        g_b = C * np.sum(func_z_y)
        return g_w, g_b

    def fit_with_logs(self, X: np.ndarray, y: np.ndarray, max_iter: int = 2000, keep_losses: bool = True) -> tuple:
        """
        Fit the model according to the given training data.

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets (+1 and -1); array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        :param keep_losses:
        :return: the training losses and accuracies during training
        """
        # Initialize learned parameters
        self.init_solution(X.shape[1])

        losses = []
        accuracies = []

        if keep_losses:
            losses.append(self.loss(self.w, self.b, self.C, X, y))
            accuracies.append(self.score(X, y))

        permutation = np.random.permutation(len(y))
        X = X[permutation, :]
        y = y[permutation]

        # Iterate over batches
        for iter in range(0, max_iter):
            start_idx = (iter * self.batch_size) % X.shape[0]
            end_idx = min(X.shape[0], start_idx + self.batch_size)
            batch_X = X[start_idx:end_idx, :]
            batch_y = y[start_idx:end_idx]

            # Compute the (sub)gradient of the current *batch*:
            g_w, g_b = self.subgradient(self.w, self.b, self.C, batch_X, batch_y)

            # Perform a (sub)gradient step
            # update the learned parameters correctly
            self.w = self.w - (self.lr * g_w)
            self.b = self.b - (self.lr * g_b)

            if keep_losses:
                losses.append(self.loss(self.w, self.b, self.C, X, y))
                accuracies.append(self.score(X, y))

        return losses, accuracies

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 2000):
        """
        Fit the model according to the given training data.
        ** Do not edit this method **

        :param X: training samples; array of shape (n_samples, n_features)
        :param y: training targets (+1 and -1); array of shape (n_samples,)
        :param max_iter: number of SGD iterations
        """
        self.fit_with_logs(X, y, max_iter=max_iter, keep_losses=False)

        return self

    def predict(self, X: np.ndarray):
        """
        Perform classification on samples in X.

        :param X: samples for prediction; array of shape (n_samples, n_features)
        :return: Predicted class labels for samples in X; array of shape (n_samples,)
                 NOTE: the labels must be either +1 or -1
        """
        # Compute the predicted labels (+1 or -1)
        margins = (X.dot(self.w) + self.b).reshape(-1, 1)
        return np.sign(margins) + (margins == 0)