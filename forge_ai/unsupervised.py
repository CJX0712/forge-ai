"""forge_ai.unsupervised —— 从零实现的无监督学习算法。

集大成：协方差特征分解复用 forge_ai.linalg.jacobi_eig；GMM 的 Σ⁻¹ 复用以
linalg.cholesky（对数稳定）。零重写：与 scikit-learn 数值交叉验证（verify.py）。

数学不变量：
- KMeans :  簇内平方和（inertia）随迭代单调不增
- PCA    :  主成分相互正交归一；秩-k 重建误差 = Eckart–Young 下界
- GMM    :  责任度每行和=1，混合权重和=1，协方差对称半正定，对数似然单调不减（EM 定理）
"""
from __future__ import annotations
import numpy as np
from . import linalg as L

__all__ = ["KMeans", "PCA", "GMM"]


def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)].copy()]
    for _ in range(k - 1):
        d = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
        probs = d / d.sum()
        centers.append(X[rng.choice(n, p=probs)].copy())
    return np.array(centers)


def _solve_lower(L, B):
    """批量前向代入：解 L Y = B（L 下三角，B 为 (d, m)）。"""
    d, m = B.shape
    Y = np.zeros((d, m))
    for i in range(d):
        Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
    return Y


def _logsumexp(a, axis=1):
    amax = a.max(axis=axis, keepdims=True)
    return amax + np.log(np.exp(a - amax).sum(axis=axis, keepdims=True))


class KMeans:
    def __init__(self, k=3, n_iter=100, seed=0, tol=1e-5):
        self.k = k
        self.n_iter = n_iter
        self.seed = seed
        self.tol = tol
        self.centers = None
        self.labels = None
        self.inertia_history_ = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        centers = _kmeanspp_init(X, self.k, rng)
        prev_inertia = np.inf
        for _ in range(self.n_iter):
            dist2 = np.stack([((X - c) ** 2).sum(axis=1) for c in centers], axis=1)
            labels = np.argmin(dist2, axis=1)
            new_centers = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                for j in range(self.k)
            ])
            inertia = dist2[np.arange(n), labels].sum()
            self.inertia_history_.append(float(inertia))
            if abs(prev_inertia - inertia) < self.tol:
                centers = new_centers
                break
            prev_inertia = inertia
            centers = new_centers
        self.centers = centers
        self.labels = np.argmin(
            np.stack([((X - c) ** 2).sum(axis=1) for c in centers], axis=1), axis=1
        )
        return self

    @property
    def inertia_monotone(self):
        h = self.inertia_history_
        return all(h[i + 1] <= h[i] + 1e-6 for i in range(len(h) - 1))


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        evals, evecs = L.jacobi_eig(cov)
        order = np.argsort(-evals)
        evals = evals[order]
        evecs = evecs[:, order]
        self.explained_variance_ = evals
        comps = evecs.T                        # 行 = 主成分
        self.explained_variance_ratio_ = evals / evals.sum()
        if self.n_components is not None:
            comps = comps[: self.n_components]
            self.explained_variance_ = evals[: self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        self.components_ = comps
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    @property
    def components_orthonormal(self):
        C = self.components_
        return float(np.max(np.abs(C @ C.T - np.eye(C.shape[0]))))


class GMM:
    """高斯混合模型 EM。

    集大成：初始化复用本库 KMeans（已收敛的 Lloyd 中心），而非单次 kmeans++ 抽样 ——
    后者在重叠簇上易把两个中心抽进同一簇，使 EM 落入较差局部最优。
    并提供 n_init 次重启取最高对数似然，与 scikit-learn 的稳健性对齐。
    """

    def __init__(self, n_components=3, n_iter=100, n_init=3, seed=0, reg=1e-6):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        self.seed = seed
        self.reg = reg
        self.weights = None
        self.means = None
        self.covs = None
        self.resp = None
        self.loglik_history_ = []

    def _em(self, X, means):
        n, d = X.shape
        K = self.n_components
        means = means.copy()
        weights = np.ones(K) / K
        covs = [np.cov(X.T) + self.reg * np.eye(d) for _ in range(K)]
        hist = []
        resp = None
        for _ in range(self.n_iter):
            # ---- E-step ----
            log_resp = np.zeros((n, K))
            for k in range(K):
                Lc = L.cholesky(covs[k])
                diff = (X - means[k]).T                  # (d, n)
                Y = _solve_lower(Lc, diff)              # (d, n)
                quad = np.sum(Y * Y, axis=0)
                logdet = 2.0 * np.sum(np.log(np.diag(Lc)))
                log_resp[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + quad) + np.log(weights[k])
            ll = _logsumexp(log_resp, axis=1)            # (n, 1)
            hist.append(float(ll.sum()))
            resp = np.exp(log_resp - ll)
            # ---- M-step ----
            Nk = resp.sum(axis=0) + 1e-10
            weights = Nk / n
            means = (resp.T @ X) / Nk[:, None]
            covs = []
            for k in range(K):
                diff = X - means[k]
                w = resp[:, k]
                cov = (w[:, None] * diff).T @ diff / Nk[k] + self.reg * np.eye(d)
                covs.append(cov)
        return weights, means, covs, resp, hist

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        best_ll = -np.inf
        best = None
        for r in range(self.n_init):
            km = KMeans(k=self.n_components, n_iter=25, seed=self.seed + r).fit(X)
            w, m, c, resp, hist = self._em(X, km.centers)
            if hist and hist[-1] > best_ll:
                best_ll = hist[-1]
                best = (w, m, c, resp, hist)
        self.weights, self.means, self.covs, self.resp, self.loglik_history_ = best
        return self

    # ---- 不变量访问器 ----
    @property
    def resp_rowsum_ok(self):
        return float(np.max(np.abs(self.resp.sum(axis=1) - 1.0)))

    @property
    def weights_sum_ok(self):
        return float(abs(self.weights.sum() - 1.0))

    @property
    def cov_symmetric_psd(self):
        worst = 0.0
        for cov in self.covs:
            worst = max(worst, float(np.max(np.abs(cov - cov.T))))
            ev, _ = L.jacobi_eig(cov)
            worst = max(worst, float(-ev.min()))           # 最小特征值应 ≥ 0
        return worst

    @property
    def loglik_monotone(self):
        h = self.loglik_history_
        return all(h[i + 1] >= h[i] - 1e-4 for i in range(len(h) - 1))
