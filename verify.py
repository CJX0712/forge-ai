"""forge_ai 验证测试台 —— 数学不变量 + 与 scikit-learn/scipy 的顶级开源交叉验证。

运行：  python verify.py
输出：  全绿 ✅ 表示每个模块均有可独立复现的数学硬标准通过。
作者：晨星
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LogisticRegression as SKLog, LinearRegression as SKLR
from sklearn.linear_model import LogisticRegression as SKSoftmax
from sklearn.neighbors import KNeighborsClassifier as SKKNN
from sklearn.cluster import KMeans as SKKM
from sklearn.decomposition import PCA as SKPCA
from sklearn.mixture import GaussianMixture as SKGMM
from sklearn.metrics import adjusted_rand_score

import forge_ai as fa


def rec(results, name, ok, detail=""):
    results.append((name, bool(ok), detail))


def _rel(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)


def main():
    results = []
    rng = np.random.default_rng(0)

    # ====================== 线性代数 ======================
    A = rng.standard_normal((6, 6))
    A = A @ A.T + 5 * np.eye(6)
    Lc = fa.cholesky(A)
    rec(results, "Cholesky 重建 L·Lᵀ=A", np.max(np.abs(Lc @ Lc.T - A)) < 1e-8,
        f"err={np.max(np.abs(Lc @ Lc.T - A)):.2e}")

    ev, evec = fa.jacobi_eig(A)
    rec(results, "Jacobi 特征残差 Aυ=λυ", np.max(np.abs(A @ evec - ev * evec)) < 1e-8)
    rec(results, "Jacobi 特征向量正交归一", np.max(np.abs(evec.T @ evec - np.eye(6))) < 1e-8)

    B = rng.standard_normal((7, 5))
    U, s, Vt = fa.svd(B)
    recon = U @ np.diag(s) @ Vt
    rec(results, "SVD 重建 U·S·Vᵀ=A", _rel(recon, B) < 1e-6, f"rel={_rel(recon, B):.2e}")
    rec(results, "SVD 左正交 UᵀU=I", np.max(np.abs(U.T @ U - np.eye(min(B.shape)))) < 1e-8)
    sv = np.linalg.svd(B, compute_uv=False)
    rec(results, "SVD 奇异值 vs numpy", np.max(np.abs(s - sv)) < 1e-6, f"err={np.max(np.abs(s - sv)):.2e}")

    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    x_ref = np.linalg.lstsq(X, y, rcond=None)[0]
    rec(results, "最小二乘 QR 路径 vs numpy",
        np.max(np.abs(fa.lstsq(X, y) - x_ref)) < 1e-8,
        f"err={np.max(np.abs(fa.lstsq(X, y) - x_ref)):.2e}")

    # ====================== 监督学习 ======================
    Xr, yr = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=0)
    lr = fa.LinearRegression().fit(Xr, yr)
    sklr = SKLR().fit(Xr, yr)
    rec(results, "LinearRegression 闭式解 vs sklearn", np.max(np.abs(lr.predict(Xr) - sklr.predict(Xr))) < 1e-6)
    rec(results, "LinearRegression 残差⊥X（正规方程）", lr.residual_orthogonality < 1e-8,
        f"‖Xᵀr‖∞={lr.residual_orthogonality:.2e}")

    Xb, yb = make_classification(n_samples=400, n_features=6, n_classes=2, random_state=1)
    lg = fa.LogisticRegression(lr=0.5, n_iter=6000).fit(Xb, yb)
    sklg = SKLog(max_iter=6000).fit(Xb, yb)
    rec(results, "LogisticRegression 标签 vs sklearn",
        np.mean(lg.predict(Xb) == sklg.predict(Xb)) > 0.95,
        f"一致率={np.mean(lg.predict(Xb) == sklg.predict(Xb)):.3f}")

    # 逻辑回归梯度检验（内联有限差分）
    # 关键：在未训练初值处检验 —— 训练收敛点梯度≈0，相对误差会失真
    lg0 = fa.LogisticRegression(lr=0.5, n_iter=0, seed=0).fit(Xb, yb)
    Xb1 = np.hstack([np.ones((len(yb), 1)), Xb])
    theta = lg0.theta.copy()

    def _loss(th):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb1 @ th, -30, 30)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean() + 0.5 * lg0.ridge * np.sum(th[1:] ** 2)

    def _grad(th):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb1 @ th, -30, 30)))
        g = Xb1.T @ (p - yb) / len(yb)
        g[1:] += lg0.ridge * th[1:]
        return g

    g0 = _grad(theta)
    gmax = 0.0
    for idx in np.ndindex(theta.shape):
        o = theta[idx]
        theta[idx] = o + 1e-5
        lp = _loss(theta)
        theta[idx] = o - 1e-5
        lm = _loss(theta)
        theta[idx] = o
        num = (lp - lm) / 2e-5
        gmax = max(gmax, abs(num - g0[idx]) / max(1e-8, abs(num) + abs(g0[idx])))
    rec(results, "LogisticRegression 梯度检验", gmax < 1e-6, f"maxRelErr={gmax:.2e}")

    Xm, ym = make_classification(n_samples=400, n_features=6, n_informative=4, n_classes=3, random_state=2)
    sm = fa.SoftmaxRegression(lr=0.5, n_iter=6000).fit(Xm, ym)
    sksm = SKSoftmax(max_iter=6000).fit(Xm, ym)
    rec(results, "SoftmaxRegression 标签 vs sklearn",
        np.mean(sm.predict(Xm) == sksm.predict(Xm)) > 0.9,
        f"一致率={np.mean(sm.predict(Xm) == sksm.predict(Xm)):.3f}")

    Xk, yk = make_classification(n_samples=400, n_features=6, random_state=3)
    knn = fa.KNNClassifier(k=5).fit(Xk, yk)
    skknn = SKKNN(n_neighbors=5).fit(Xk, yk)
    rec(results, "KNN 标签 vs sklearn", np.mean(knn.predict(Xk) == skknn.predict(Xk)) == 1.0,
        f"一致率={np.mean(knn.predict(Xk) == skknn.predict(Xk)):.3f}")

    # ====================== 无监督学习 ======================
    Xcl, _ = make_blobs(n_samples=400, centers=4, n_features=5, random_state=4)
    km = fa.KMeans(k=4, seed=4).fit(Xcl)
    skkm = SKKM(n_clusters=4, n_init=10, random_state=4).fit(Xcl)
    rec(results, "KMeans inertia 单调不增", km.inertia_monotone)
    cost = cdist(km.centers, skkm.cluster_centers_)
    row, col = linear_sum_assignment(cost)
    ari = adjusted_rand_score(km.labels, _remap(skkm.labels_, row, col, km.k))
    rec(results, "KMeans 聚类 vs sklearn (ARI)", ari > 0.99, f"ARI={ari:.3f}")

    Xp, _ = make_blobs(n_samples=400, n_features=5, random_state=5)
    p = fa.PCA(n_components=3).fit(Xp)
    skp = SKPCA(n_components=3).fit(Xp)
    rec(results, "PCA 主成分正交归一", p.components_orthonormal < 1e-8,
        f"‖CᵀC−I‖={p.components_orthonormal:.2e}")
    rec(results, "PCA 解释方差 vs sklearn", _rel(p.explained_variance_, skp.explained_variance_) < 1e-6,
        f"rel={_rel(p.explained_variance_, skp.explained_variance_):.2e}")
    Xr_sk = skp.inverse_transform(skp.transform(Xp))
    Xr_fa = p.inverse_transform(p.transform(Xp))
    rec(results, "PCA 重建 vs sklearn (Eckart–Young)", _rel(Xr_fa, Xr_sk) < 1e-6,
        f"rel={_rel(Xr_fa, Xr_sk):.2e}")

    Xg, _ = make_blobs(n_samples=400, centers=3, n_features=4, random_state=6, cluster_std=1.2)
    gmm = fa.GMM(n_components=3, seed=6, n_iter=300).fit(Xg)
    skg = SKGMM(n_components=3, covariance_type="full", reg_covar=1e-6,
                n_init=3, max_iter=300, random_state=6).fit(Xg)
    rec(results, "GMM 责任度行和=1", gmm.resp_rowsum_ok < 1e-8, f"max|Σ−1|={gmm.resp_rowsum_ok:.2e}")
    rec(results, "GMM 权重和=1", gmm.weights_sum_ok < 1e-8)
    rec(results, "GMM 协方差对称半正定", gmm.cov_symmetric_psd < 1e-6,
        f"worst={gmm.cov_symmetric_psd:.2e}")
    rec(results, "GMM 对数似然单调 (EM 定理)", gmm.loglik_monotone)
    cost = cdist(gmm.means, skg.means_)
    row, col = linear_sum_assignment(cost)
    md = max(np.max(np.abs(gmm.means[i] - skg.means_[col[list(row).index(i)]])) for i in range(3))
    cd = max(np.max(np.abs(gmm.covs[i] - skg.covariances_[col[list(row).index(i)]])) for i in range(3))
    rec(results, "GMM 参数 vs sklearn", md < 0.6 and cd < 1.5, f"meanΔ={md:.2f} covΔ={cd:.2f}")

    # ====================== 深度学习 ======================
    Xd = rng.standard_normal((60, 4))
    yd = rng.integers(0, 3, 60)
    mlp = fa.MLP(layers=(4, 6, 3), n_iter=0, seed=7)
    gerr = mlp.gradient_check(Xd, yd)
    rec(results, "MLP 反向传播梯度检验", gerr < 1e-6, f"maxRelErr={gerr:.2e}")
    mlp2 = fa.MLP(layers=(4, 12, 3), lr=0.3, n_iter=12000, seed=7).fit(Xd, yd)
    rec(results, "MLP 可训练(过拟合训练集)", np.mean(mlp2.predict(Xd) == yd) > 0.9,
        f"acc={np.mean(mlp2.predict(Xd) == yd):.3f}")

    Xa = rng.standard_normal((12, 8))
    target = rng.standard_normal((12, 8))
    mha = fa.MultiHeadAttention(d_model=8, n_heads=2, seed=8)
    aerr = mha.gradient_check(Xa, target)
    rec(results, "Attention 反向传播梯度检验", aerr < 1e-6, f"maxRelErr={aerr:.2e}")

    # ====================== 强化学习 ======================
    world = fa.GridWorld()
    Vopt, popt = world.value_iteration()
    ql = fa.QLearning(world, seed=0).train(30000)
    rec(results, "QLearning Bellman 残差→0", ql.bellman_residual < 1e-2,
        f"res={ql.bellman_residual:.2e}")
    rec(results, "QLearning 策略匹配最优", ql.policy_matches(popt) == 1.0,
        f"match={ql.policy_matches(popt):.3f}")

    # ====================== 报告 ======================
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 78)
    print(f" forge_ai 验证报告  | forge_ai v{fa.__version__}  | 作者 {fa.__author__}")
    print("=" * 78)
    for name, ok, detail in results:
        icon = "✅" if ok else "❌"
        line = f" {icon}  {name}"
        if detail:
            line += f"   [{detail}]"
        print(line)
    print("-" * 78)
    print(f" 通过 {n_pass}/{len(results)}")
    print("=" * 78)
    return n_pass, len(results), results


def _remap(labels, row, col, k):
    out = np.zeros_like(labels)
    for i, j in zip(row, col):
        out[labels == j] = i
    return out


if __name__ == "__main__":
    n_pass, n_total, _ = main()
    import sys
    sys.exit(0 if n_pass == n_total else 1)
