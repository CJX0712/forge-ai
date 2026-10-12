# 架构与数据流

作者：晨星

## 1. 分层

```
┌─────────────────────────────────────────────────────────┐
│  用户 API          verify.py / js/forge.js 自检           │
├─────────────────────────────────────────────────────────┤
│  算法层   supervised │ unsupervised │ deep │ rl          │
│           Linear  Logistic  Softmax  kNN                │
│           KMeans  PCA  GMM                              │
│           MLP   Attention  MultiHeadAttention           │
│           GridWorld  QLearning                          │
├─────────────────────────────────────────────────────────┤
│  内核层   linalg：Cholesky │ Jacobi 特征 │ 单侧 Jacobi   │
│                   SVD │ QR │ 最小二乘                    │
├─────────────────────────────────────────────────────────┤
│  原语层   numpy.ndarray（BLAS 级张量存储与矩阵乘）         │
└─────────────────────────────────────────────────────────┘
```

依赖是**严格单向**的：上层 → `linalg` → numpy。没有环。

## 2. 内核复用关系（集大成的具体体现）

| 上层模块 | 复用的内核 | 用途 |
| --- | --- | --- |
| `LinearRegression` | `solve_psd`（Cholesky） | 正规方程闭式解 `(XᵀX+λI)⁻¹Xᵀy` |
| `lstsq` | `qr` | 最小二乘 QR 路径 |
| `PCA` | `jacobi_eig` | 协方差矩阵特征分解 |
| `GMM` | `cholesky` + 前向代入 | `Σ⁻¹(x−μ)` 与 `logdet`（数值稳定，不用 `slogdet`） |
| `GMM`（初始化） | `KMeans` | Lloyd 已收敛中心，替代单次 kmeans++ 抽样 |
| `MLP` / `Attention` | 纯 matmul | 前向与反向 |

## 3. 数据流：GMM 一次 EM 迭代

```
X (n×d)
  │
  ├─ E 步：对每个 k
  │    L = cholesky(Σₖ)
  │    Y = 前向代入(L, (X−μₖ)ᵀ)      # 解 L·Y = (X−μₖ)ᵀ
  │    quad = Σ Y²                    # = (x−μ)ᵀΣ⁻¹(x−μ)
  │    logdet = 2·Σ log diag(L)
  │    log_resp[:,k] = −½(d·log2π + logdet + quad) + log πₖ
  │
  ├─ logsumexp 稳定化 → llᵢ = log Σₖ exp(log_respᵢₖ)
  │    resp = exp(log_resp − ll)
  │
  └─ M 步：πₖ = Nₖ/n,  μₖ = Σᵢ γᵢₖxᵢ/Nₖ,  Σₖ = Σᵢ γᵢₖ(xᵢ−μₖ)(xᵢ−μₖ)ᵀ/Nₖ + reg·I
       │
       └──▶ 断言：ll 单调不减（EM 定理）
```

稳定化要点：全程在对数空间计算，`logsumexp` 防溢出；协方差加 `reg·I` 保证正定，
Cholesky 才有解。

## 4. 数据流：多头注意力一次前向

```
X (n×d_model)
  │
  ├─ Q = X·Wq,  K = X·Wk,  V = X·Wv
  │
  ├─ 分头：Qh = Q.reshape(n,H,d_k).transpose(1,0,2)
  │
  ├─ 每头：scores = Qh·Khᵀ/√d ─▶ softmax ─▶ out_h = A·Vh
  │
  ├─ 拼接：concat = stack(outs, axis=1).reshape(n, H·d_k)   ← 逐样本拼头
  │
  └─ out = concat · Wo
```

> ⚠️ 拼接顺序是本项目踩过的最隐蔽的坑：
> `stack(axis=1).transpose(1,0,2).reshape()` 得到的是"按时间拼接"，
> 梯度检验会直接给 `maxRelErr = 1.0`。

## 5. 复现性

- 所有随机源都用 `numpy.random.default_rng(seed)`，种子写死在 `verify.py`。
- JS 内核用 `mulberry32` 自实现 RNG（不依赖 `Math.random`），保证跨引擎一致。
- `verify.py` 全程无网络、无 GPU 依赖，退出码 `0` 表示全绿。

## 6. 扩展一个新算法的 checklist

1. 在 `forge_ai/` 下新建模块，**只依赖 `linalg`**（禁止直接调 `numpy.linalg` 分解）。
2. 为算法写一条**数学不变量**，以 `@property` 暴露（如 `resp_rowsum_ok`）。
3. 在 `verify.py` 加两条记录：不变量 + 与 sklearn/numpy 的交叉验证。
4. 同步在 `docs/INVARIANTS.md` 补表。
5. 跑 `python verify.py`，必须全绿才可提交。

---

作者：晨星
