# API 参考

作者：晨星

```python
import forge_ai as fa
```

---

## `forge_ai.linalg`

| 函数 | 说明 |
| --- | --- |
| `cholesky(A)` | Cholesky 分解，返回下三角 `L`，`A = L·Lᵀ` |
| `cholesky_solve(L, b)` | 解 `L·Lᵀx = b`（前向 + 后向代入） |
| `solve_psd(A, b)` | 解对称正定系统（Cholesky 路径） |
| `jacobi_eig(A, max_sweeps=200, tol=1e-12)` | 对称矩阵 Jacobi 特征分解，返回 `(特征值升序, 特征向量列)` |
| `svd(A, max_sweeps=50, tol=1e-12)` | 单侧 Jacobi SVD，返回 `(U, s, Vt)` |
| `qr(A)` | 修正 Gram–Schmidt QR，返回 `(Q, R)` |
| `lstsq(A, b)` | 最小二乘（QR 路径） |

---

## `forge_ai.supervised`

### `LinearRegression(ridge=0.0)`
- `fit(X, y)` —— 闭式解 `(XᵀX + λI)⁻¹Xᵀy`，Cholesky 路径
- `predict(X)`
- `residual_orthogonality` —— 不变量 `‖Xᵀr‖∞`（应 ≈ 0）

### `LogisticRegression(lr=0.3, n_iter=3000, ridge=1e-6, seed=0)`
- `fit(X, y)` / `predict(X)` / `predict_proba(X)`
- 截距不参与正则化

### `SoftmaxRegression(lr=0.3, n_iter=3000, ridge=1e-6, seed=0)`
- `fit(X, y)`（y 为 0..K-1 整数标签）/ `predict(X)` / `predict_proba(X)`

### `KNNClassifier(k=3)`
- `fit(X, y)` / `predict(X)`（暴力 L2）

---

## `forge_ai.unsupervised`

### `KMeans(k=3, n_iter=100, seed=0, tol=1e-5)`
- `fit(X)` —— kmeans++ 初始化 + Lloyd
- 属性：`centers`, `labels`, `inertia_history_`, `inertia_monotone`

### `PCA(n_components=None)`
- `fit(X)` / `transform(X)` / `inverse_transform(Z)`
- 属性：`mean_`, `components_`, `explained_variance_`,
  `explained_variance_ratio_`, `components_orthonormal`

### `GMM(n_components=3, n_iter=100, n_init=3, seed=0, reg=1e-6)`
- `fit(X)` —— Lloyd 初始化 + EM，`n_init` 次重启取最高对数似然
- 属性：`weights`, `means`, `covs`, `resp`, `loglik_history_`
- 不变量：`resp_rowsum_ok`, `weights_sum_ok`, `cov_symmetric_psd`, `loglik_monotone`

---

## `forge_ai.deep`

### `MLP(layers=(4, 8, 3), lr=0.1, n_iter=2000, seed=0)`
- 隐层 `tanh`，输出 `softmax`，交叉熵损失
- `fit(X, y)` / `predict(X)` / `gradient_check(X, y, eps=1e-5)` → `maxRelErr`

### `scaled_dot_product_attention(Q, K, V, mask=None)`
- 返回 `(out, cache)`

### `MultiHeadAttention(d_model, n_heads, seed=0)`
- `forward(X)` → `(out, cache)`
- `gradient_check(X, target, eps=1e-5)` → `maxRelErr`（覆盖 `Wq/Wk/Wv/Wo`）

---

## `forge_ai.rl`

### `GridWorld(n=4, goal=(3,3), step_cost=-1.0, gamma=0.99)`
- `step(s, a)` → `(s', reward, done)`
- `value_iteration(tol=1e-9, max_iter=1000)` → `(V, policy)`（oracle）

### `QLearning(world, lr=0.1, eps=0.15, seed=0)`
- `train(n_episodes=8000)`
- `bellman_residual`（屏蔽终点态）/ `greedy_policy()` / `policy_matches(optimal_policy)`

---

## `js/forge.js`

```js
const F = require('./js/forge.js');   // 或浏览器全局 ForgeAI
F.selfTest();   // → [{name, ok, detail}, ...]
```

导出：`cholesky`, `jacobiEig`, `svd`, `pcaFit`, `pcaTransform`, `pcaInverse`,
`gmmFit`, `mlpForward`, `mlpBackward`, `mlpLoss`, `attention`,
`attentionBackward`, `knnPredict`, `selfTest`。

命令行：`node js/forge.js` —— 全绿退出码 `0`。

---

作者：晨星
