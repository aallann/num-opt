import abc
import numpy as np
import pandas as pd

from utils import Callback
from handlers import F


class Optimizer(abc.ABC):
    """Abstract base optimizer class"""

    def __init__(self):
        super().__init__()
        pass

    @abc.abstractmethod
    def optimize(self):
        """Optimize function"""
        return NotImplementedError


class svmOptimiser(Optimizer):
    """Semi-smooth Newton optimiser for SVM"""

    def __init__(
        self,
        x: np.array,
        y: np.array,
        sigma: float = 0.2,
        delta: float = 1e-9,
        eta: float = 0.05,
        kappa: float = 0.3,
        C: float = None,
        rho: float = 0.9,
        maxIter: int = 1000,
        w_init: np.array = False,
    ):
        super().__init__()

        # data
        self.x = x
        self.y = y

        # parameters
        self.sigma = sigma
        self.delta = delta
        self.eta = eta
        self.kappa = kappa
        self.C = C
        self.rho = rho
        self.maxIter = maxIter
        self.stop_condition = True
        self.sIter = 0
        self.cgIter = 0

        if not w_init:
            w_init = np.ones(x.shape[1])

        self.w = w_init
        self.data = pd.DataFrame(
            columns=["Iteration", "Objective", "Gradient Norm", "Active Set Size", "w"]
        )
        self.alphas = []

    def optimize(
        self,
        f: F,
        callback: Callback = None,
    ):
        """implements semismooth newton optimization"""
        while self.stop_condition and self.sIter < self.maxIter:
            self.stop_condition: bool = (
                np.linalg.norm(f.gradient(self.w, self.x, self.y, self.C), ord=2)
                > self.delta
            )

            if self.sIter > 1 and False:
                if callback:
                    self.stop_condition: bool = (
                        self.data["Active Set Size"].iloc[-2]
                        - self.data["Active Set Size"].iloc[-1]
                        > 0
                    )

            g: np.array = f.gradient(self.w, self.x, self.y, self.C)
            V: np.array = f.V(
                self.w, self.x, self.y, f.active_set(self.w, self.x, self.y), self.C
            )
            d: np.array = np.ones(self.w.shape[0])
            r: np.array = -g - np.dot(V, d)
            p: np.array = r
            psi: float = np.min([self.eta, self.kappa * np.linalg.norm(g)])

            cg_stop_condition: bool = True
            cg_maxIter: int = 200
            j = 0
            while cg_stop_condition and j < cg_maxIter:

                if psi * np.linalg.norm(g) > np.linalg.norm(r):
                    cg_stop_condition = False

                alpha: float = np.dot(r.T, r) / np.dot(p.T, np.dot(V, p))
                d = d + alpha * p  # update
                r_ = r
                r = r - alpha * np.dot(V, p)

                if psi * np.linalg.norm(g) > np.linalg.norm(r):
                    cg_stop_condition = False

                beta: float = np.dot(r.T, r - r_) / np.dot((r - r_).T, p)
                p = r + beta * p  # update

                j += 1

            # cg iter end
            self.cgIter += j

            alpha = 0.5
            k = 0
            while f.objective(self.w + alpha * d, self.x, self.y, self.C) > f.objective(
                self.w, self.x, self.y, self.C
            ) + self.sigma * alpha * np.dot(g.T, d):
                k += 1

            # line search end
            self.w = self.w + alpha**k * d
            self.alphas.append(alpha)

            if callback:
                callback.update_state_dict(
                    active_set_size=f.active_set(self.w, self.x, self.y).shape[0],
                    objective=f.objective(self.w, self.x, self.y, self.C),
                    w=self.w,
                    iter=self.sIter,
                    gradient_norm=np.linalg.norm(
                        f.gradient(self.w, self.x, self.y, self.C)
                    ),
                )

            self.data.loc[self.sIter, :] = [
                self.sIter,
                f.objective(self.w, self.x, self.y, self.C),
                np.linalg.norm(g),
                len(f.active_set(self.w, self.x, self.y)),
                self.w,
            ]

            self.sIter += 1

        return self.data, self.w, self.sIter, self.cgIter, self.alphas
