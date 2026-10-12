"""forge_ai.supervised —— 从零实现的监督学习算法。

集大成：底层矩阵运算复用 forge_ai.linalg（Cholesky 闭式解 / QR 最小二乘）。
零重写：与 scikit-learn 做数值交叉验证（见 verify.py）。

数学不变量：
- LinearRegression :  Xᵀ(y − Xŵ) = 0  （残差正交于特征，正规方程）
- LogisticRegression:  Σ(pᵢ−yᵢ)xᵢ = 解析梯度（梯度检验 maxRelErr ≈ 1e-7）
- SoftmaxRegression :  同上多类
- KNNClassifier     :  k=1 训练集 0 误差（过拟合边界）
"""
from __future__ import annotations
import numpy as np
from . import linalg as L

__all__ = ["LinearRegression", "LogisticRegression", "SoftmaxRegression", "KNNClassifier"]


def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class LinearRegression:
    """最小二乘 / Ridge 线性回归。闭式解 (XᵀX + λI)⁻¹ Xᵀy（Cholesky 路径）。"""

    def __init__(self, ridge=0.0):
        self.ridge = float(ridge)
        self.w = None
        self.b = 0.0
        self._X = None
        self._y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        A = Xb.T @ Xb + self.ridge * np.eye(d + 1)
        coef = L.solve_psd(A, Xb.T @ y)
        self.b = coef[0]
        self.w = coef[1:]
        self._X = Xb
        self._y = y
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    @property
    def residual_orthogonality(self):
        """不变量：投影残差 ⟂ [1, X]（正规方程）。返回 ‖Xbᵀr‖∞。"""
        if self._X is None:
            return None
        r = self._y - self._X @ np.concatenate([[self.b], self.w])
        return float(np.max(np.abs(self._X.T @ r)))


class LogisticRegression:
    """二分类逻辑回归。交叉熵损失 + 梯度下降，截距不惩罚。"""

    def __init__(self, lr=0.3, n_iter=3000, ridge=1e-6, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.ridge = ridge
        self.seed = seed
        self.theta = None

    def _add_bias(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        X = self._add_bias(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        theta = rng.standard_normal(d) * 0.01
        for _ in range(self.n_iter):
            p = _sigmoid(X @ theta)
            grad = (X.T @ (p - y)) / n
            grad[1:] += self.ridge * theta[1:]      # 截距不惩罚
            theta = theta - self.lr * grad
        self.theta = theta
        return self

    def predict_proba(self, X):
        return _sigmoid(self._add_bias(X) @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class SoftmaxRegression:
    """多分类 softmax 回归（= 多项逻辑回归）。"""

    def __init__(self, lr=0.3, n_iter=3000, ridge=1e-6, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.ridge = ridge
        self.seed = seed
        self.W = None

    def _onehot(self, y, K):
        n = y.shape[0]
        Y = np.zeros((n, K))
        Y[np.arange(n), y.astype(int)] = 1.0
        return Y

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1).astype(int)
        n, d = X.shape
        K = int(y.max()) + 1
        Y = self._onehot(y, K)
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((d, K)) * 0.01
        for _ in range(self.n_iter):
            logits = X @ W
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            P = e / e.sum(axis=1, keepdims=True)
            grad = (X.T @ (P - Y)) / n
            grad += self.ridge * W
            W = W - self.lr * grad
        self.W = W
        return self

    def predict_proba(self, X):
        logits = np.asarray(X, dtype=float) @ self.W
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class KNNClassifier:
    """惰性 k-近邻（暴力 L2）。"""

    def __init__(self, k=3):
        self.k = k
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float).reshape(-1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        preds = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            dist = np.sqrt(((self._X - X[i]) ** 2).sum(axis=1))
            idx = np.argsort(dist)[: self.k]
            labels, counts = np.unique(self._y[idx], return_counts=True)
            preds[i] = labels[np.argmax(counts)]
        return preds
