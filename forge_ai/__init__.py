"""forge_ai —— 世界级 AI 算法内核（从零手写 · 数学不变量可验证 · 与顶级开源交叉验证）。

作者：晨星
"""
from .linalg import (
    matmul, transpose, add, sub, eye, cholesky, cholesky_solve,
    solve_psd, jacobi_eig, svd, qr, lstsq,
)
from .supervised import (
    LinearRegression, LogisticRegression, SoftmaxRegression, KNNClassifier,
)
from .unsupervised import KMeans, PCA, GMM
from .deep import MLP, scaled_dot_product_attention, MultiHeadAttention
from .rl import GridWorld, QLearning

__version__ = "1.0.0"
__author__ = "晨星"

__all__ = [
    "matmul", "transpose", "add", "sub", "eye", "cholesky", "cholesky_solve",
    "solve_psd", "jacobi_eig", "svd", "qr", "lstsq",
    "LinearRegression", "LogisticRegression", "SoftmaxRegression", "KNNClassifier",
    "KMeans", "PCA", "GMM", "MLP", "scaled_dot_product_attention",
    "MultiHeadAttention", "GridWorld", "QLearning",
]
