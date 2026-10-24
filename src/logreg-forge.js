/*!
 * logreg-forge.js — 晨星旗舰 AI · L1 监督学习：逻辑回归
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · 交叉熵损失 + 批量梯度下降 —— Nelder & Wedderburn 1972（GLM 框架）
 *   · 最大似然 = 最小化 NLL（统计推断金标准）
 *   · 数值稳定 sigmoid（分段计算，杜绝 exp 溢出）
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 终止一阶条件   ‖∇w‖ ≤ 1e-4（凸优化全局最优 ⇒ 梯度归零）
 *   [2] 训练准确率     ≥ 0.95
 *   [3] 概率合法性     p ∈ (0,1)（距边界有正边际）
 *   [4] 损失收敛       末段均值 ≤ 首段均值（凸性保证单调下降）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LogRegForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const LG = {};

  LG.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  LG.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  /* 数值稳定 sigmoid：z≥0 走 1/(1+e^{-z})，z<0 走 e^z/(1+e^z)，双分支均不溢出 */
  LG.sigmoid = function (z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  };

  LG.fit = function (X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.5, iters = opts.iters || 20000, lambda = opts.lambda || 0;
    const n = X.length, d = X[0].length;
    const w = new Array(d).fill(0);
    const Xt = [];
    for (let j = 0; j < d; j++) { Xt.push([]); for (let i = 0; i < n; i++) Xt[j].push(X[i][j]); }
    const hist = [];
    for (let it = 0; it < iters; it++) {
      const p = X.map(function (row) {
        let z = 0;
        for (let j = 0; j < d; j++) z += row[j] * w[j];
        return LG.sigmoid(z);
      });
      for (let j = 0; j < d; j++) {
        let g = 0;
        for (let i = 0; i < n; i++) g += (p[i] - y[i]) * X[i][j];
        w[j] -= lr * (g / n + lambda * w[j]);
      }
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const pi = Math.min(Math.max(p[i], 1e-12), 1 - 1e-12);
        loss += -(y[i] * Math.log(pi) + (1 - y[i]) * Math.log(1 - pi));
      }
      hist.push(loss / n);
    }
    return { w: w, history: hist };
  };

  LG.predictProba = function (X, w) {
    return X.map(function (row) {
      let z = 0;
      for (let j = 0; j < w.length; j++) z += row[j] * w[j];
      return LG.sigmoid(z);
    });
  };

  /* ================= 自检 ================= */
  LG.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = LG.mulberry32(20261024);
    const X = [], y = [];
    for (let c = 0; c < 2; c++) {
      const mu = c === 0 ? -1.5 : 1.5;
      for (let i = 0; i < 40; i++) {
        X.push([1, mu + LG.randn(rng), mu + LG.randn(rng)]);
        y.push(c);
      }
    }
    // L2 正则使目标强凸；lr=0.05 满足 GD 稳定条件 0<lr<2/λ_max(Hessian)
    const fit = LG.fit(X, y, { lr: 0.05, iters: 60000, lambda: 0.01 });
    const w = fit.w;
    const p = LG.predictProba(X, w);

    // [1] 终止一阶条件：全梯度 = 数据梯度 + λw（强凸 ⇒ 唯一最优解处严格归零）
    let gNorm = 0;
    for (let j = 0; j < w.length; j++) {
      let g = 0;
      for (let i = 0; i < X.length; i++) g += (p[i] - y[i]) * X[i][j];
      gNorm = Math.max(gNorm, Math.abs(g / X.length + 0.01 * w[j]));
    }
    add('终止一阶条件 max|∇w|', gNorm, 1e-5);

    // [2] 训练准确率
    let correct = 0;
    for (let i = 0; i < X.length; i++) if ((p[i] >= 0.5 ? 1 : 0) === y[i]) correct++;
    add('训练准确率', correct / X.length, 0.95, 'ge');

    // [3] 概率合法性：距 (0,1) 边界的正边际
    let minP = 1, maxP = 0;
    for (const v of p) { if (v < minP) minP = v; if (v > maxP) maxP = v; }
    add('概率合法边际 min(p,1−p)', Math.min(minP, 1 - maxP), 1e-12, 'ge');

    // [4] 损失收敛：末段均值 ≤ 初始值（w=0 时 = ln2）的 10%
    const h = fit.history;
    const avg = function (arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; };
    add('损失收敛（末500均值/初始）', avg(h.slice(-500)) / h[0], 0.1);
    add('最终交叉熵损失', h[h.length - 1], 0.2);

    return { name: 'logreg-forge', layer: 'L1 监督·逻辑回归', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return LG;
});
