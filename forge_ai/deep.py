"""forge_ai.deep —— 从零实现的深度学习原语（MLP 反向传播 + 注意力机制）。

集大成：纯 numpy 存储，矩阵乘复用 BLAS；梯度计算从零推导。
零重写：与手写解析梯度的*有限差分*交叉验证（梯度检验 maxRelErr ≈ 1e-7）。

数学不变量（可验证硬标准）：
- MLP          :  maxRelErr(∇数值, ∇解析) < 1e-6
- Attention    :  同上，反向传播解析梯度与中心差分一致
"""
from __future__ import annotations
import numpy as np

__all__ = ["MLP", "scaled_dot_product_attention", "MultiHeadAttention"]


def _softmax(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=axis, keepdims=True)


# ============================ MLP ============================
class MLP:
    """全连接前馈网络：隐层 tanh，输出 softmax，交叉熵损失。从零前向+反向。"""

    def __init__(self, layers=(4, 8, 3), lr=0.1, n_iter=2000, seed=0):
        self.layers = layers
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((layers[i], layers[i + 1])) * 0.1
                  for i in range(len(layers) - 1)]
        self.b = [np.zeros(layers[i + 1]) for i in range(len(layers) - 1)]

    def _forward(self, X):
        a = np.asarray(X, dtype=float)
        zs, acts = [], [a]
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = np.tanh(z)
            zs.append(z)
            acts.append(a)
        z = a @ self.W[-1] + self.b[-1]
        p = _softmax(z, axis=-1)
        zs.append(z)
        acts.append(p)
        return p, acts, zs

    def _loss(self, X, y):
        p, _, _ = self._forward(X)
        n = X.shape[0]
        ll = -np.log(p[np.arange(n), y.astype(int)] + 1e-12).mean()
        return float(ll)

    def _backward(self, X, y):
        p, acts, zs = self._forward(X)
        n = X.shape[0]
        Y = np.zeros_like(p)
        Y[np.arange(n), y.astype(int)] = 1.0
        grads = {"W": [None] * len(self.W), "b": [None] * len(self.b)}
        # 输出层：softmax+CE → dz = p - Y
        dz = p - Y
        grads["W"][-1] = acts[-2].T @ dz / n
        grads["b"][-1] = dz.mean(axis=0)
        # 反向至隐层
        for i in range(len(self.W) - 2, -1, -1):
            da = dz @ self.W[i + 1].T
            dz = da * (1.0 - acts[i + 1] ** 2)        # tanh' = 1 - a²
            grads["W"][i] = acts[i].T @ dz / n
            grads["b"][i] = dz.mean(axis=0)
        return grads

    def fit(self, X, y):
        for _ in range(self.n_iter):
            g = self._backward(X, y)
            for i in range(len(self.W)):
                self.W[i] -= self.lr * g["W"][i]
                self.b[i] -= self.lr * g["b"][i]
        return self

    def predict(self, X):
        p, _, _ = self._forward(X)
        return np.argmax(p, axis=1)

    def gradient_check(self, X, y, eps=1e-5):
        g = self._backward(X, y)
        max_rel = 0.0
        for key in ("W", "b"):
            for i in range(len(self.W)):
                arr = self.W[i] if key == "W" else self.b[i]
                grad = g[key][i]
                for idx in np.ndindex(arr.shape):
                    orig = arr[idx]
                    arr[idx] = orig + eps
                    lp = self._loss(X, y)
                    arr[idx] = orig - eps
                    lm = self._loss(X, y)
                    arr[idx] = orig
                    num = (lp - lm) / (2 * eps)
                    ana = grad[idx]
                    denom = max(1e-8, abs(num) + abs(ana))
                    max_rel = max(max_rel, abs(num - ana) / denom)
        return float(max_rel)


# ====================== 注意力机制 ======================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力。返回 out 与反向所需 cache。"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if mask is not None:
        scores = scores + mask
    A = _softmax(scores, axis=-1)
    out = A @ V
    return out, (A, Q, K, V, d)


def _attention_backward(dOut, cache):
    A, Q, K, V, d = cache
    dV = A.T @ dOut
    dA = dOut @ V.T
    s = (dA * A).sum(axis=-1, keepdims=True)
    dscores = dA * A - s * A                      # softmax 雅可比
    dQ = dscores @ K / np.sqrt(d)
    dK = dscores.T @ Q / np.sqrt(d)
    return dQ, dK, dV


class MultiHeadAttention:
    """多头注意力（从零投影 + 拼接 + 输出投影），含梯度检验。"""

    def __init__(self, d_model, n_heads, seed=0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.default_rng(seed)
        std = 0.1
        self.Wq = rng.standard_normal((d_model, d_model)) * std
        self.Wk = rng.standard_normal((d_model, d_model)) * std
        self.Wv = rng.standard_normal((d_model, d_model)) * std
        self.Wo = rng.standard_normal((d_model, d_model)) * std

    def _split(self, M):
        n = M.shape[0]
        return M.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        Qh = self._split(Q)
        Kh = self._split(K)
        Vh = self._split(V)
        outs = []
        caches = []
        for h in range(self.n_heads):
            o, c = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h])
            outs.append(o)
            caches.append(c)
        # 拼接：stack→(n,H,d_k) 再 reshape→(n,H*d_k)，即逐样本拼接各头
        concat = np.stack(outs, axis=1).reshape(X.shape[0], self.d_model)
        out = concat @ self.Wo
        return out, (X, Qh, Kh, Vh, outs, concat, caches)

    def _loss(self, X, target):
        out, _ = self.forward(X)
        return float(0.5 * np.sum((out - target) ** 2))

    def gradient_check(self, X, target, eps=1e-5):
        out, (Xr, Qh, Kh, Vh, outs, concat, caches) = self.forward(X)
        dOut = out - target
        # 反向：输出投影
        dConcat = dOut @ self.Wo.T
        dWo = concat.T @ dOut
        # 还原各头梯度
        dConcat_heads = dConcat.reshape(X.shape[0], self.n_heads, self.d_k).transpose(1, 0, 2)
        dQh = np.zeros_like(Qh)
        dKh = np.zeros_like(Kh)
        dVh = np.zeros_like(Vh)
        for h in range(self.n_heads):
            dq, dk, dv = _attention_backward(dConcat_heads[h], caches[h])
            dQh[h], dKh[h], dVh[h] = dq, dk, dv
        dQ_full = dQh.transpose(1, 0, 2).reshape(X.shape[0], self.d_model)
        dK_full = dKh.transpose(1, 0, 2).reshape(X.shape[0], self.d_model)
        dV_full = dVh.transpose(1, 0, 2).reshape(X.shape[0], self.d_model)
        analytic = {
            "Wq": X.T @ dQ_full,
            "Wk": X.T @ dK_full,
            "Wv": X.T @ dV_full,
            "Wo": dWo,
        }
        max_rel = 0.0
        for name in ("Wq", "Wk", "Wv", "Wo"):
            arr = getattr(self, name)
            grad = analytic[name]
            for idx in np.ndindex(arr.shape):
                orig = arr[idx]
                arr[idx] = orig + eps
                lp = self._loss(X, target)
                arr[idx] = orig - eps
                lm = self._loss(X, target)
                arr[idx] = orig
                num = (lp - lm) / (2 * eps)
                denom = max(1e-8, abs(num) + abs(grad[idx]))
                max_rel = max(max_rel, abs(num - grad[idx]) / denom)
        return float(max_rel)
