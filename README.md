# forge-ai · 世界级 AI 算法内核

> **从零手写的算法 × 数学不变量可验证 × 与顶级开源交叉验证**
>
> 作者：晨星（GitHub [CJX0712](https://github.com/CJX0712)） · MIT License
>
> ```text
> Python 主库   verify.py      →  28 / 28  ✅   （与 numpy / scikit-learn 机器级对齐）
> JS 单文件核   node js/forge.js  →  11 / 11  ✅
> JS 分模块内核 node src/test-runner.js → 60 / 60 ✅（10 模块，浏览器/无头双端一致）
> ```

`forge-ai` 是一套**可证明正确**的 AI 算法内核：线性代数 → 监督学习 → 无监督学习 → 深度学习 → 强化学习，每个算法都从零手写，并且每一条都配一条**可被独立实现交叉验证的数学不变量**（硬金标准）。

它不是又一个玩具库，而是一份**能被证伪的教科书**：任何一条不变量挂了，就说明实现错了。

两条并行轨道，同一套不变量体系：

| 轨道 | 位置 | 特点 | 验证 |
| --- | --- | --- | --- |
| **Python 主库** | `forge_ai/` | 以 numpy 作张量原语，与 sklearn / scipy 逐项对齐 | `python verify.py` → 28 项 |
| **JS 单文件核** | `js/forge.js` | 零依赖、无头 Node 自检 | `node js/forge.js` → 11 项 |
| **JS 分模块内核** | `src/*.js` | 十层架构 L0–L6 逐层锻造，单文件自包含，浏览器一键自检 | `node src/test-runner.js` → 60 项 |

---

## 一、设计原则

| 原则 | 落地方式 |
| --- | --- |
| **零重写** | 不重复造轮子。以 `numpy` 作张量原语（BLAS 级），以 `scikit-learn` / `scipy` 作交叉验证 oracle；本库只从零实现**算法本身**。 |
| **集大成** | 每个模块站在顶级成果之上：单侧 Jacobi SVD（Hestenes 1960）、Cholesky、EM 定理、Bellman 方程、Eckart–Young 定理。GMM 初始化复用本库已收敛的 Lloyd（KMeans），而非单次 kmeans++ 抽样。 |
| **可验证** | 每条算法配一条数学不变量；`verify.py` 一次性跑完 28 项（Python）；`src/test-runner.js` 跑完 60 项（JS 分模块）。 |
| **闭环交付** | 可运行、可复现、可验证，全部开源，署名晨星。 |

---

## 二、为什么"从零手写"不算造轮子

关键在于**验证链**：

```text
你的实现  ──数学不变量──▶  硬标准（解析解 / 定理）
    │
    └──数值对齐──▶  numpy.linalg / scikit-learn（世界顶级实现）
```

- 与 sklearn **完全一致**：LinearRegression 闭式解、KNN、KMeans(ARI=1.000)、
  GMM(meanΔ=0.00, covΔ=0.00)、PCA(rel=5e-15)。
- 与 numpy **机器级一致**：SVD 奇异值误差 `1.78e-15`，Cholesky 重建 `1.78e-15`。
- 与自身定理**自洽**：EM 对数似然单调、KMeans inertia 单调、Bellman 残差 → 0。

---

## 三、十层分解架构

JS 分模块内核（`src/`）按十层组织，依赖方向单向、能力逐层叠加：

```text
L9 应用与评测        ← 面向任务的编排、SOP、基准
L8 大模型系统        ← 推理引擎、缓存、调度
L7 生成模型          ← 分布建模与采样
L6 强化学习          ← 序贯决策
L5 序列 / 注意力     ← 上下文建模
L4 深度学习          ← 可微函数逼近
L3 概率图模型        ← 结构化概率推断
L2 无监督学习        ← 结构发现
L1 监督学习          ← 映射学习
L0 数学内核          ← 数值与线性代数
```

| 层 | 主题 | 状态 | 内容 |
| --- | --- | --- | --- |
| L0 | 数学内核 | ✅ | Jacobi 特征分解、Cholesky 求解、可复现随机、中心差分梯度检验 |
| L1 | 监督学习 | ✅ | Ridge 线性回归、逻辑回归、CART 决策树 |
| L2 | 无监督学习 | ✅ | PCA、K-Means、GMM/EM |
| L3 | 概率图模型 | ○ | 朴素贝叶斯 / HMM（Viterbi、前向后向）规划中 |
| L4 | 深度学习 | ✅ | MLP 反向传播 + 梯度检验 |
| L5 | 序列 / 注意力 | ✅ | 缩放点积注意力、多头机制、softmax Jacobian 反向 |
| L6 | 强化学习 | ✅ | Q-Learning（Bellman 残差验证） |
| L7 | 生成模型 | ○ | VAE / GAN / 扩散模型 规划中 |
| L8 | 大模型系统 | ○ | Tokenizer、KV Cache、推理调度 规划中 |
| L9 | 应用与评测 | ○ | RAG、Agent 编排、1024 项微指令级 SOP 分解 规划中 |

分层规范见 [`docs/ARCHITECTURE_JS.md`](docs/ARCHITECTURE_JS.md)；Python 主库的架构与数据流见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 四、快速开始

```bash
# ① Python 主库（核心仅需 numpy；验证台另需 scikit-learn + scipy）
pip install numpy scikit-learn scipy
python verify.py                    # → 28 项不变量 + 交叉验证

# ② JS 单文件内核（零依赖）
node js/forge.js                    # → 11 项自检

# ③ JS 分模块内核（零依赖，十层 L0–L6）
node src/test-runner.js             # → 10 模块 / 60 项，退出码 0 = 全绿

# ④ 浏览器一键自检（无需构建、无需服务器）
#    直接打开 index.html，本地完成全部计算，零网络请求
```

### Python 用法示例

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

### JS 用法示例

```js
const MathForge = require('./src/math-forge.js');

// 对称矩阵特征分解：解析解交叉验证
const { values } = MathForge.jacobiEigenSym([[2, 1], [1, 2]]);
// values → [3, 1]（误差 8.88e-16）

// 梯度检验：解析梯度 vs 中心差分
const err = MathForge.gradientCheck(
  x => x.reduce((s, v) => s + v * v, 0),
  x => x.map(v => 2 * v),
  [0.7, -1.3, 2.1]
);
// err → ≈ 1e-11
```

浏览器环境：每个模块挂载到 `window`（`window.MathForge`、`window.NNForge`…），`<script src="src/math-forge.js"></script>` 即可使用。

---

## 五、验证结果（实测）

### ① Python — `python verify.py` → **28 / 28 ✅**

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

### ② JS 单文件核 — `node js/forge.js` → **11 / 11 ✅**

```text
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

### ③ JS 分模块内核 — `node src/test-runner.js` → **10 模块 / 60 项 ✅ 全绿**

| 文件 | 算法 | 核心不变量 | 检查数 |
| --- | --- | --- | --- |
| `src/math-forge.js` | 线性代数 / 数值内核 | `‖VᵀV−I‖=1.78e-15`、`‖Av−λv‖=4.88e-15`、`‖LLᵀ−A‖=1.78e-15` | 10 |
| `src/linreg-forge.js` | 线性回归（Ridge） | 闭式 vs 梯度下降一致 `2.89e-8`、正规方程残差 `9.93e-11` | 5 |
| `src/logreg-forge.js` | 逻辑回归 | 强凸 ⇒ 梯度归零 `2.21e-15`、准确率 1.0 | 5 |
| `src/tree-forge.js` | CART 决策树 | 分裂增益 ≥ 0、叶子 Gini = 0 | 5 |
| `src/pca-forge.js` | 主成分分析 | 谱守恒 `5.43e-16`、行列式守恒 `1.58e-15`、得分去相关 `1.69e-14` | 9 |
| `src/kmeans-forge.js` | K-Means | SSE 单调不增违例 `0`、纯度 1.0 | 6 |
| `src/gmm-forge.js` | GMM / EM | 似然单调违例 `3.41e-13`、模型选择优于单高斯 | 6 |
| `src/nn-forge.js` | MLP + 反向传播 | 17 参数梯度检验 `1.03e-9`、XOR 100% | 6 |
| `src/attn-forge.js` | 注意力 / 多头 | 64 参数梯度检验 `1.82e-8`、softmax 行和 `2.22e-16` | 4 |
| `src/qlearn-forge.js` | Q-Learning | 84 对 Bellman 残差 `4.98e-6`、最优路径 8 步 | 4 |

浏览器侧 `index.html` 使用**同一份** `selfTest()`，保证 Node 与浏览器两条执行路径结论一致。

---

## 六、验证方法学

每个算法按下列优先级建立不变量（优先级越高越难作弊）：

| 优先级 | 方法 | 例子 |
| --- | --- | --- |
| 1 | 解析解对照 | Jacobi λ=`{3,1}`、PCA 特征值 `{25,1,0.04}` |
| 2 | 双独立路径互证 | 闭式 vs 梯度下降；Cholesky det vs 特征值乘积 |
| 3 | 定理单调性 | SSE 单调、EM 似然单调、凸损失单调下降 |
| 4 | 一阶/最优性条件 | `‖∇w‖→0`、Bellman 残差→0、正规方程残差=0 |
| 5 | 梯度检验 | 解析梯度 vs 中心差分（反向传播正确性的唯一金标准） |
| 6 | 守恒律 | 概率归一（softmax 行和、责任度行和、权重和） |

---

## 七、踩过的坑（都写进注释了）

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
7. **逻辑回归学习率需满足 GD 稳定条件**：数据列相关使 Hessian 的 `λ_max≈10`，
   `lr=0.5` 超过 `2/λ_max` 导致大特征方向震荡。降到 `0.05` 后梯度归零至 `2.2e-15`。
8. **Q-Learning 低频状态-动作对会残留陈旧偏差**：常数学习率下罕访问的对永远收敛不到位
   （残差 0.87）。引入 **exploring starts** + **末期纯随机刷新段**后，84 对残差降到 `4.98e-6`。

---

## 八、目录

```text
forge-ai/
├── README.md                本文件（双轨统一入口）
├── LICENSE                  MIT，作者晨星
├── requirements.txt
├── pyproject.toml
├── verify.py                Python 统一验证台（28 项）
├── forge_ai/                Python 内核
│   ├── linalg.py            数值线性代数（Cholesky / Jacobi / 单侧 Jacobi SVD / QR）
│   ├── supervised.py        LinearRegression · LogisticRegression · SoftmaxRegression · kNN
│   ├── unsupervised.py      KMeans · PCA · GMM(EM)
│   ├── deep.py              MLP（反向传播）· 缩放点积注意力 · 多头注意力
│   ├── rl.py                GridWorld（价值迭代 oracle）· QLearning
│   └── __init__.py
├── js/forge.js              单文件零依赖 JS 内核（11 项自检）
├── src/                     JS 分模块内核（十层 L0–L6，60 项自检）
│   ├── math-forge.js        L0 数学内核
│   ├── linreg-forge.js      L1 线性回归
│   ├── logreg-forge.js      L1 逻辑回归
│   ├── tree-forge.js        L1 CART 决策树
│   ├── pca-forge.js         L2 PCA
│   ├── kmeans-forge.js      L2 K-Means
│   ├── gmm-forge.js         L2 GMM / EM
│   ├── nn-forge.js          L4 MLP
│   ├── attn-forge.js        L5 注意力 / Transformer
│   ├── qlearn-forge.js      L6 Q-Learning
│   └── test-runner.js       无头自检总控（exit 0/1）
├── index.html               浏览器一键自检驾驶舱
└── docs/
    ├── ARCHITECTURE.md      Python 主库架构与数据流
    ├── ARCHITECTURE_JS.md   JS 分模块内核十层架构与接口契约
    ├── INVARIANTS.md        数学不变量全景（公式 + 判定）
    └── API.md               API 参考
```

---

## 九、许可

MIT License。作者：**晨星（CJX0712）**。

---

*锻造世界顶级 AI：架构最先进、性能最强、代码最优、文档最全。*
