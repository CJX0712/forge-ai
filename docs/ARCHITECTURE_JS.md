# 晨星旗舰 AI · 十层分解架构（JS 分模块内核）

> 本文档定义 `src/` 下 JS 分模块内核的架构分层、接口契约与扩展规范。
> Python 主库（`forge_ai/`）的架构与数据流见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；
> 数学不变量全景见 [`INVARIANTS.md`](INVARIANTS.md)。
>
> 作者：晨星（CJX0712） · MIT License

---

## 一、分层原则

架构按**「依赖方向单向、能力逐层叠加」**组织：上层可以调用下层，下层绝不感知上层。

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

**关键不变量**：任意一层单独运行其 `selfTest()` 必须全绿，且不依赖任何上层模块。

---

## 二、模块接口契约

每个 `*-forge.js` 必须满足：

```js
// 1. UMD 双通道导出
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XXXForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  return {
    /* ...算法 API... */

    // 2. 自检必须存在且返回统一结构
    selfTest: function () {
      return {
        name: 'xxx-forge',        // 模块名（= 文件名）
        layer: 'L2 无监督·PCA',    // 所属层
        pass: true,               // 全部检查是否通过
        checks: [
          {
            name: '正交归一 ‖VᵀV−I‖',   // 不变量名（人类可读，含数学式）
            value: 4.44e-16,            // 实测值
            tol:   1e-14,               // 容差
            mode:  'le',                // 'le' = value ≤ tol；'ge' = value ≥ tol
            ok:    true                 // 判定结果
          }
        ]
      };
    }
  };
});
```

### 新增模块的接入步骤

1. 在 `src/` 下新建 `xxx-forge.js`，遵循上述契约；
2. 在 `src/test-runner.js` 的 `MODULES` 数组中追加 `['层名', './xxx-forge.js']`；
3. 在 `index.html` 中追加 `<script src="src/xxx-forge.js"></script>` 与 `MODULES` 注册项；
4. 运行 `node src/test-runner.js`，确保退出码为 `0`；
5. 在 `README.md` 的模块清单表中补一行（算法 / 参考文献 / 核心不变量 / 检查数）。

---

## 三、自检总控的工作方式

`src/test-runner.js` 的判定逻辑**完全交给模块自身**（`r.pass`），总控只做编排与汇总：

```text
require 模块 → 调用 selfTest() → 汇总 pass / checks → 打印表格 → exit(0|1)
```

- 退出码 `0`：全部模块、全部检查项通过（CI 友好）
- 退出码 `1`：存在失败，并打印失败明细（不变量名 + 实测值 + 容差）

浏览器侧 `index.html` 使用同一份 `selfTest()`，保证**Node 与浏览器两条执行路径结论一致**。

---

## 四、L0 数学内核的能力边界

| 能力 | API | 复杂度 | 数值特性 |
| --- | --- | --- | --- |
| 对称特征分解 | `jacobiEigenSym(A)` | O(n³)/sweep | 绝对收敛；黄金标准精度，适合 n ≤ 200 |
| SPD 直接求解 | `cholesky(A)` / `solveSPD(A,b)` | O(n³/3) | 半带宽内最优；不显式求逆（数值稳定性关键） |
| 可复现随机 | `mulberry32(seed)` / `randn(rng)` | O(1) | 同种子 ⇒ 同随机流 |
| 梯度检验 | `gradientCheck(f, g, x, eps)` | O(n) 次 f 求值 | 中心差分，理论误差 O(ε²) |

> **为什么不用幂迭代 / QR？** Jacobi 在中小规模下精度最高（特征向量正交性可达 1e-15 量级），且实现无分支陷阱，是「写一遍就能信一辈子」的选择。规模上去后可替换为分治 QR，接口不变。

---

## 五、第十层（L9）微指令分解

L9 按企业级 SOP 拆解为 **1024 项微指令级操作**，按 4 个域组织：

| 域 | 微指令数 | 说明 |
| --- | --- | --- |
| 数据域 | 256 | 采集 / 清洗 / 标注 / 切片 / 版本 |
| 模型域 | 256 | 训练 / 微调 / 蒸馏 / 量化 / 部署 |
| 推理域 | 256 | 路由 / 检索 / 工具调用 / 后处理 |
| 评测域 | 256 | 指标 / 回归 / 对齐 / 安全 |

每项微指令须满足：**输入可枚举、输出可断言、失败可定位**——与本库「硬不变量」的方法论一脉相承。

---

## 六、扩展指南：把「更强的现成方案」替换进来

当出现更优实现时，替换路径应保持**不变量不变**：

1. 保留 `selfTest()` 的检查项集合（必要时新增，不删除）；
2. 替换内部实现（如 Jacobi → 分治 QR、全批量 GD → AdamW）；
3. 重跑 `node src/test-runner.js`，验证不变量仍成立；
4. 在文件头注释中标注新的参考文献与替换理由。

> 零重写原则的落点就在这里：**接口是骨架，实现是肌肉，不变量是神经。**

---

*锻造世界顶级 AI：架构最先进、性能最强、代码最优、文档最全。*
