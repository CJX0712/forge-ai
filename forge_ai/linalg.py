"""forge_ai.linalg —— 从零实现的数值线性代数内核。

设计原则（集大成 + 零重写）：
- 以 numpy 的 ndarray 作为张量存储与基础运算原语（BLAS 级顶级实现），
- 但所有*分解算法*（Cholesky / Jacobi 特征分解 / SVD / QR）均从零手写，
- 并以 numpy.linalg 作为 oracle 做交叉验证（见 verify.py）。

数学不变量（可验证硬标准）：
- Cholesky :  L @ L.T == A            （残差 < 1e-10）
- Jacobi    :  A v == λ v ，特征向量相互正交归一
- SVD       :  U @ diag(S) @ Vt == A ，UᵀU==I，Vt Vtᵀ==I，S 与 np.linalg.svd 一致
- QR        :  Q @ R == A ，QᵀQ == I
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "matmul", "transpose", "add", "sub", "eye", "cholesky",
    "cholesky_solve", "solve_psd", "jacobi_eig", "svd", "qr", "lstsq",
]


# ---------- 基础运算（委托 BLAS 级原语） ----------
def matmul(A, B):
    return np.dot(A, B)


def transpose(A):
    return np.asarray(A, dtype=float).T


def add(A, B):
    return np.asarray(A, dtype=float) + np.asarray(B, dtype=float)


def sub(A, B):
    return np.asarray(A, dtype=float) - np.asarray(B, dtype=float)


def eye(n):
    return np.eye(n)


# ---------- Cholesky 分解 A = L Lᵀ（对称正定） ----------
def cholesky(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                d = A[i, i] - s
                if d <= 0:                      # 数值保护：处理近似 PSD
                    d = max(d, 1e-12)
                L[i, i] = np.sqrt(d)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def cholesky_solve(L, b):
    """解 L Lᵀ x = b（前向代入 + 后向代入）。"""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, k] * y[k] for k in range(i))) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k, i] * x[k] for k in range(i + 1, n))) / L[i, i]
    return x


def solve_psd(A, b):
    """解对称正定系统 A x = b（Cholesky 路径）。"""
    return cholesky_solve(cholesky(A), b)


# ---------- 经典循环 Jacobi 特征分解（对称矩阵） ----------
def jacobi_eig(A, max_sweeps=200, tol=1e-12):
    """返回 (eigenvalues 升序, eigenvectors 列向量)。"""
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += A[p, q] ** 2
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                if abs(A[p, p] - A[q, q]) < 1e-300:
                    t = 1.0 if A[p, q] >= 0 else -1.0      # App==Aqq → 45° 旋转
                else:
                    theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                    t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                App, Aqq, Apq = A[p, p], A[q, q], A[p, q]
                A[p, p] = c * c * App - 2 * s * c * Apq + s * s * Aqq
                A[q, q] = s * s * App + 2 * s * c * Apq + c * c * Aqq
                A[p, q] = 0.0
                A[q, p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        Akp, Akq = A[k, p], A[k, q]
                        A[k, p] = c * Akp - s * Akq
                        A[k, q] = s * Akp + c * Akq
                        A[p, k] = A[k, p]
                        A[q, k] = A[k, q]
                for k in range(n):
                    Vkp, Vkq = V[k, p], V[k, q]
                    V[k, p] = c * Vkp - s * Vkq
                    V[k, q] = s * Vkp + c * Vkq
    eigvals = np.diag(A)
    idx = np.argsort(eigvals)
    return eigvals[idx], V[:, idx]


# ---------- 单侧 Jacobi SVD：A = U diag(S) Vt（从零，Hestenes 1960） ----------
def svd(A, max_sweeps=50, tol=1e-12):
    """通过右旋列正交化（one-sided Jacobi）直接对 A 做 SVD，精度达机器级。"""
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += (A[:, p] @ A[:, q]) ** 2
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                Ap = A[:, p].copy()
                Aq = A[:, q].copy()
                gamma = Ap @ Aq
                if abs(gamma) < 1e-300:
                    continue
                alpha = Ap @ Ap
                beta = Aq @ Aq
                zeta = (beta - alpha) / (2.0 * gamma)
                t = np.sign(zeta) / (abs(zeta) + np.sqrt(zeta * zeta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                A[:, p] = c * Ap - s * Aq
                A[:, q] = s * Ap + c * Aq
                Vp = V[:, p].copy()
                Vq = V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
    sigma = np.array([np.linalg.norm(A[:, j]) for j in range(n)])
    order = np.argsort(-sigma)
    sigma = sigma[order]
    V = V[:, order]
    k = min(m, n)
    U = np.zeros((m, k))
    for j in range(k):
        U[:, j] = A[:, order[j]] / max(sigma[j], 1e-12)
    Q, _ = qr(U)                            # 数值正交化兜底
    return Q, sigma[:k], V[:, :k].T


# ---------- 修正 Gram–Schmidt QR 分解 ----------
def qr(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] if R[j, j] > 0 else 0.0
    return Q, R


# ---------- 最小二乘 A x ≈ b（QR 路径） ----------
def lstsq(A, b):
    Q, R = qr(A)
    b = np.asarray(b, dtype=float).reshape(-1)
    y = Q.T @ b
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(R[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / R[i, i] if R[i, i] != 0 else 0.0
    return x
