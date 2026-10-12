# forge-ai · 世界级 AI 算法内核

> **从零手写的算法 × 数学不变量可验证 × 与顶级开源交叉验证**
>
> 作者：晨星（GitHub [CJX0712](https://github.com/CJX0712)）

`forge-ai` 是一套**可证明正确**的 AI 算法内核：线性代数 → 监督学习 → 无监督学习 →
深度学习 → 强化学习，每个算法都从零手写，并且每一条都配一条**可被独立实现交叉验证的
数学不变量**（硬金标准），最终与 scikit-learn / numpy / scipy 逐项对齐。

它不是又一个玩具库，而是一份**能被证伪的教科书**：任何一条不变量挂了，就说明实现错了。

---

## 一、设计原则

| 原则 | 落地方式 |
| --- | --- |
| **零重写** | 不重复造轮子。以 `numpy` 作张量原语（BLAS 级），以 `scikit-learn` / `scipy` 作交叉验证 oracle；本库只从零实现**算法本身**。 |
| **集大成** | 每个模块站在顶级成果之上：单侧 Jacobi SVD（Hestenes 1960）、Cholesky、EM 定理、Bellman 方程、Eckart–Young 定理。GMM 初始化复用本库已收敛的 Lloyd（KMeans），而非单次 kmeans++ 抽样。 |
| **可验证** | 每条算法配一条数学不变量；`verify.py` 一次性跑完 28 项（Python）+ 11 项（JS）。 |
| **闭环交付** | 可运行、可复现、可验证，全部开源，署名晨星。 |

---

## 二、为什么"从零手写"不算造轮子

关键在于**验证链**：

```
你的实现  ──数学不变量──▶  硬标准（解析解 / 定理）
    │
    └──数值对齐──▶  numpy.linalg / scikit-learn（世界顶级实现）
```

- 与 sklearn **完全一致**：LinearRegression 闭式解、KNN、KMeans(ARI=1.000)、
  GMM(meanΔ=0.00, covΔ=0.00)、PCA(rel=5e-15)。
- 与 numpy **机器级一致**：SVD 奇异值误差 `1.78e-15`，Cholesky 重建 `1.78e-15`。
- 与自身定理**自洽**：EM 对数似然单调、KMeans inertia 单调、Bellman 残差 → 0。

---

## 三、架构

```
forge_ai/
├── linalg.py       数值线性代数内核（Cholesky / Jacobi 特征 / 单侧 Jacobi SVD / QR）
│                   └─ 被所有上层模块复用（闭式解、协方差分解、Σ⁻¹）
├── supervised.py   LinearRegression · LogisticRegression · SoftmaxRegression · kNN
├── unsupervised.py KMeans · PCA · GMM(EM)
├── deep.py         MLP（反向传播）· 缩放点积注意力 · 多头注意力
├── rl.py           GridWorld（价值迭代 oracle）· QLearning
└── __init__.py

verify.py           统一验证台：28 项不变量 + 交叉验证
js/forge.js          单文件零依赖 JS 内核，无头自检 11 项
```

依赖倒置清晰：上层**只依赖** `linalg`，`linalg` 只依赖 numpy。

---

## 四、快速开始

```bash
# 核心库仅需 numpy；验证台额外需要 scikit-learn + scipy
pip install numpy scikit-learn scipy

# Python 验证（28 项）
python verify.py

# JS 内核自检（11 项，零依赖）
node js/forge.js
```

### 用法示例

```python
import numpy as np
import forge_ai as fa

# 线性回归：闭式解 + 残差正交不变量
lr = fa.LinearRegression().fit(X, y)
print(lr.residual_orthogonality)      # ‖Xᵀr‖∞ ≈ 6e-12

# PCA：主成分正交 + Eckart–Young 重建
p = fa.PCA(n_components=3).fit(X)
Z = p.transform(X)
Xr = p.inverse_transform(Z)
print(p.components_orthonormal)       # ≈ 8.9e-16

# GMM：EM 全部不变量
g = fa.GMM(n_components=3, n_init=3).fit(X)
print(g.resp_rowsum_ok, g.weights_sum_ok, g.cov_symmetric_psd, g.loglik_monotone)

# 多头注意力：梯度检验
mha = fa.MultiHeadAttention(d_model=8, n_heads=2)
print(mha.gradient_check(X, target))  # maxRelErr ≈ 2e-7
```

---

## 五、验证结果（实测）

### Python — `python verify.py` → **28 / 28 ✅**

| 模块 | 检查项 | 结果 |
| --- | --- | --- |
| 线性代数 | Cholesky 重建 `L·Lᵀ=A` | `err=1.78e-15` |
| | Jacobi 特征残差 `Aυ=λυ` / 正交归一 | ✅ |
| | SVD 重建 `U·S·Vᵀ=A` | `rel=1.01e-15` |
| | SVD 奇异值 vs `numpy.linalg.svd` | `err=1.78e-15` |
| | 最小二乘 QR 路径 vs `numpy.linalg.lstsq` | `err=1.67e-16` |
| 监督学习 | LinearRegression 闭式解 vs sklearn | ✅ |
| | LinearRegression 残差⊥X（正规方程） | `‖Xᵀr‖∞=5.98e-12` |
| | LogisticRegression 标签 vs sklearn | 一致率 `1.000` |
| | LogisticRegression 梯度检验 | `maxRelErr=6.98e-09` |
| | SoftmaxRegression 标签 vs sklearn | 一致率 `0.917` |
| | kNN 标签 vs sklearn | 一致率 `1.000` |
| 无监督学习 | KMeans inertia 单调不增 | ✅ |
| | KMeans 聚类 vs sklearn | `ARI=1.000` |
| | PCA 主成分正交归一 | `8.88e-16` |
| | PCA 解释方差 vs sklearn | `rel=5.26e-15` |
| | PCA 重建 vs sklearn（Eckart–Young） | `rel=3.10e-07` |
| | GMM 责任度行和=1 / 权重和=1 | `9.99e-16` |
| | GMM 协方差对称半正定 | `4.16e-17` |
| | GMM 对数似然单调（EM 定理） | ✅ |
| | GMM 参数 vs sklearn | `meanΔ=0.00 covΔ=0.00` |
| 深度学习 | MLP 反向传播梯度检验 | `maxRelErr=1.25e-08` |
| | MLP 可训练（过拟合训练集） | `acc=1.000` |
| | Attention 反向传播梯度检验 | `maxRelErr=1.98e-07` |
| 强化学习 | QLearning Bellman 残差→0 | `res=5.37e-04` |
| | QLearning 策略匹配最优 | `match=1.000` |

### JS — `node js/forge.js` → **11 / 11 ✅**

```
✅ Cholesky 重建 L·Lᵀ=A                    [err=3.55e-15]
✅ Jacobi 特征残差 Aυ=λυ                    [err=7.11e-15]
✅ Jacobi 特征向量正交归一                   [err=5.55e-16]
✅ SVD 重建 U·S·Vᵀ=A                       [rel=5.85e-16]
✅ PCA 主成分正交归一                        [err=6.66e-16]
✅ PCA 重建误差 = Eckart–Young 下界          [rel=2.65e-15]
✅ GMM 责任度行和=1 & 权重和=1               [resp=4.4e-16]
✅ GMM 对数似然单调 (EM 定理)
✅ MLP 反向传播梯度检验                      [maxRelErr=2.49e-9]
✅ Attention 反向传播梯度检验                 [maxRelErr=5.19e-8]
✅ kNN k=1 训练集零误差（过拟合边界）         [acc=40/40]
```

---

## 六、踩过的坑（都写进注释了）

1. **SVD 用 `AᵀA` 特征分解会平方放大条件数** → 6×6 随机矩阵重建误差掉到 `1e-7`。
   改用**单侧 Jacobi**（Hestenes）直接对 A 列正交化，奇异值精度回到 `1e-15`。
2. **numpy 切片是视图**：Jacobi 里 `Ap = A[:,p]` 后原地赋值，`Ap` 已被污染，
   第二轮旋转用了脏数据 → SVD 完全崩坏。改用 `.copy()`。
3. **多头拼接顺序错乱**：`stack(axis=1).transpose(1,0,2).reshape()` 拼出的是
   "按时间拼接"而非"按头拼接"，梯度检验 `maxRelErr=1.0`。正确写法是
   `stack(axis=1).reshape(n, H*d_k)`。
4. **梯度检验要在未训练初值处做**：训练收敛点梯度≈0，相对误差分母趋零导致误报
   （曾报 `1.66e-04`，实为假阳性）。
5. **Bellman 残差必须屏蔽终点态**：Q-learning 到终点即终止，终点态 Q 恒为 0，
   把终点态算进残差会恒定得到 `1.0`。
6. **GMM 单次 kmeans++ 抽样**可能把两个中心抽进同一簇 → EM 落入较差局部最优
   （LL −3302 vs 最优 −2925）。改 Lloyd 初始化 + `n_init` 重启后与 sklearn 完全一致。

---

## 七、目录

```
forge-ai/
├── README.md
├── LICENSE                 MIT，作者晨星
├── requirements.txt
├── pyproject.toml
├── verify.py               统一验证台
├── forge_ai/               Python 内核
│   ├── linalg.py
│   ├── supervised.py
│   ├── unsupervised.py
│   ├── deep.py
│   ├── rl.py
│   └── __init__.py
├── js/forge.js             单文件零依赖 JS 内核
└── docs/
    ├── ARCHITECTURE.md     架构与数据流
    ├── INVARIANTS.md       数学不变量全景（公式 + 判定）
    └── API.md              API 参考
```

---

## 八、许可

MIT License。作者：**晨星**。
