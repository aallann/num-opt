import numpy as np
import pandas as pd


class F(object):
    """function handler for SVM optimisation"""

    def __init__(self):
        pass

    def objective(w: np.ndarray, x: np.ndarray, y: np.ndarray, C: float) -> float:
        """objective function."""
        return 0.5 * np.linalg.norm(w, ord=2) ** 2 + C * np.sum(
            np.maximum(0, 1 - y * np.dot(x, w)) ** 2
        )

    def gradient(w: np.ndarray, x: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
        """gradient of the objective function."""
        return w - 2 * C * np.sum(
            np.multiply(np.maximum(0, 1 - y * np.dot(x, w)) * y, x.T).T, axis=0
        )

    def active_set(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> list:
        """active set function"""
        return np.where(1 - y * np.dot(x, w) > 0)[0]

    def V(
        w: np.ndarray, x: np.ndarray, y: np.ndarray, active: list, C: float
    ) -> np.ndarray:
        """generalised jacobian space, V"""
        return np.eye(len(w)) + 2 * C * x[active].T @ x[active]


class FPolyKernel(object):
    """function handler for SVM optimisation"""

    def __init__(self, gamma: float = -0.0011, coef0: int = 0, degree: int = 2):
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def poly_kernel(self, x1, x2):
        """Polynomial kernel function"""
        return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree

    def objective(
        self,
        w: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        C: float,
    ) -> float:
        """objective function."""
        return 0.5 * np.linalg.norm(w, ord=2) ** 2 + C * np.sum(
            np.maximum(0, 1 - y * np.array([self.poly_kernel(xi, w) for xi in x])) ** 2
        )

    def gradient(
        self, w: np.ndarray, x: np.ndarray, y: np.ndarray, C: float
    ) -> np.ndarray:
        """gradient of the objective function."""
        return w - 2 * C * np.sum(
            np.multiply(
                np.maximum(0, 1 - y * np.array([self.poly_kernel(xi, w) for xi in x]))
                * y,
                x.T,
            ).T,
            axis=0,
        )

    def active_set(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> list:
        """active set function"""
        return np.where(1 - y * np.array([self.poly_kernel(xi, w) for xi in x]) > 0)[0]

    def V(
        self, w: np.ndarray, x: np.ndarray, y: np.ndarray, active: list, C: float
    ) -> np.ndarray:
        """generalised jacobian space, V"""
        return np.eye(len(w)) + 2 * C * np.array(
            [self.poly_kernel(xi, w) for xi in x[active]]
        ).T @ np.array([self.poly_kernel(xi, w) for xi in x[active]])


class FGaussianKernel(object):
    """function handler for SVM optimisation"""

    def __init__(self, gamma: float = 1 / 9):
        self.gamma = gamma

    def rbf_kernel(self, x1, x2):
        """RBF kernel function"""
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def objective(
        self,
        w: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        C: float,
    ) -> float:
        """objective function."""
        return 0.5 * np.linalg.norm(w, ord=2) ** 2 + C * np.sum(
            np.maximum(0, 1 - y * np.array([self.rbf_kernel(xi, w) for xi in x])) ** 2
        )

    def gradient(
        self, w: np.ndarray, x: np.ndarray, y: np.ndarray, C: float
    ) -> np.ndarray:
        """gradient of the objective function."""
        return w - 2 * C * np.sum(
            np.multiply(
                np.maximum(0, 1 - y * np.array([self.rbf_kernel(xi, w) for xi in x]))
                * y,
                x.T,
            ).T,
            axis=0,
        )

    def active_set(self, w: np.ndarray, x: np.ndarray, y: np.ndarray) -> list:
        """active set function"""
        return np.where(1 - y * np.array([self.rbf_kernel(xi, w) for xi in x]) > 0)[0]

    def V(
        self, w: np.ndarray, x: np.ndarray, y: np.ndarray, active: list, C: float
    ) -> np.ndarray:
        """generalised jacobian space, V"""
        return np.eye(len(w)) + 2 * C * np.array(
            [self.rbf_kernel(xi, w) for xi in x[active]]
        ).T @ np.array([self.rbf_kernel(xi, w) for xi in x[active]])
