/*!
 * nn-forge.js — 晨星旗舰 AI · L4 深度学习：多层感知机（MLP）
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · 反向传播 —— Rumelhart, Hinton & Williams 1986（链式法则的工程化身）
 *   · tanh / sigmoid 激活 + 二元交叉熵
 *   · 全批量梯度下降（凸性不保证 ⇒ 用梯度检验锁定实现正确性）
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 梯度检验      解析梯度 vs 中心差分 maxRelErr ≤ 1e-6（反向传播正确性的唯一金标准）
 *   [2] XOR 求解      非线性可分任务 100% 正确（万能逼近定理的最小实证）
 *   [3] 损失收敛      末段均值 ≤ 初始值 10%
 *   [4] 正确类边际    min p(correct) ≥ 0.5
 *   [5] 训练稳定性    max|W| ≤ 20（无梯度爆炸）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NNForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const NN = {};

  NN.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  NN.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const HID = 4; // 隐层宽度

  /* 参数扁平化：[W1(4×2) | b1(4) | W2(1×4) | b2(1)]，长度 17 */
  function unflatten(p) {
    return {
      W1: [p.slice(0, 2), p.slice(2, 4), p.slice(4, 6), p.slice(6, 8)],
      b1: p.slice(8, 12),
      W2: [p.slice(12, 16)],
      b2: [p[16]]
    };
  }
  function flatten(P) {
    return [].concat(P.W1[0], P.W1[1], P.W1[2], P.W1[3], P.b1, P.W2[0], P.b2);
  }

  function tanh(z) { return Math.tanh(z); }

  NN.forward = function (P, X) {
    const A1 = [], Z1 = [];
    for (let s = 0; s < X.length; s++) {
      const z1 = [], a1 = [];
      for (let h = 0; h < HID; h++) {
        let z = P.b1[h];
        for (let j = 0; j < 2; j++) z += P.W1[h][j] * X[s][j];
        z1.push(z); a1.push(tanh(z));
      }
      Z1.push(z1); A1.push(a1);
    }
    const A2 = [];
    for (let s = 0; s < X.length; s++) {
      let z = P.b2[0];
      for (let h = 0; h < HID; h++) z += P.W2[0][h] * A1[s][h];
      A2.push(1 / (1 + Math.exp(-z)));
    }
    return { A1: A1, Z1: Z1, A2: A2 };
  };

  NN.loss = function (P, X, y) {
    const F = NN.forward(P, X);
    let l = 0;
    for (let s = 0; s < X.length; s++) {
      const p = Math.min(Math.max(F.A2[s], 1e-12), 1 - 1e-12);
      l += -(y[s] * Math.log(p) + (1 - y[s]) * Math.log(1 - p));
    }
    return l / X.length;
  };

  /* 反向传播：返回扁平梯度 */
  NN.gradients = function (P, X, y) {
    const F = NN.forward(P, X);
    const n = X.length;
    const dW1 = [], db1 = [], dW2 = [0, 0, 0, 0], db2 = [0];
    for (let h = 0; h < HID; h++) { dW1.push([0, 0]); db1.push(0); }
    for (let s = 0; s < n; s++) {
      const dz2 = (F.A2[s] - y[s]) / n;
      db2[0] += dz2;
      for (let h = 0; h < HID; h++) dW2[h] += dz2 * F.A1[s][h];
      for (let h = 0; h < HID; h++) {
        const da1 = dz2 * P.W2[0][h];
        const dz1 = da1 * (1 - F.A1[s][h] * F.A1[s][h]); // tanh′(z)=1−a²
        db1[h] += dz1;
        for (let j = 0; j < 2; j++) dW1[h][j] += dz1 * X[s][j];
      }
    }
    return flatten({ W1: dW1, b1: db1, W2: [dW2], b2: db2 });
  };

  NN.train = function (X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.5, iters = opts.iters || 20000, seed = opts.seed || 42;
    const rng = NN.mulberry32(seed);
    const P = {
      W1: [], b1: [], W2: [[]], b2: [0]
    };
    for (let h = 0; h < HID; h++) {
      P.W1.push([NN.randn(rng), NN.randn(rng)]);
      P.b1.push(0);
    }
    for (let h = 0; h < HID; h++) P.W2[0].push(NN.randn(rng) * 0.5);
    let p = flatten(P);
    const hist = [];
    for (let it = 0; it < iters; it++) {
      const g = NN.gradients(unflatten(p), X, y);
      for (let i = 0; i < p.length; i++) p[i] -= lr * g[i];
      hist.push(NN.loss(unflatten(p), X, y));
    }
    return { params: unflatten(p), history: hist };
  };

  /* ================= 自检 ================= */
  NN.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const y = [0, 1, 1, 0];

    // [1] 梯度检验（随机初值，全 17 参数）
    const rng = NN.mulberry32(20261024);
    const P0 = {
      W1: [[NN.randn(rng), NN.randn(rng)], [NN.randn(rng), NN.randn(rng)],
           [NN.randn(rng), NN.randn(rng)], [NN.randn(rng), NN.randn(rng)]],
      b1: [0.1, -0.2, 0.3, -0.1],
      W2: [[NN.randn(rng), NN.randn(rng), NN.randn(rng), NN.randn(rng)]],
      b2: [0.2]
    };
    const f = function (pv) { return NN.loss(unflatten(pv), X, y); };
    const g = function (pv) { return NN.gradients(unflatten(pv), X, y); };
    let maxRel = 0;
    const p0 = flatten(P0);
    for (let i = 0; i < p0.length; i++) {
      const eps = 1e-5;
      const pp = p0.slice(), pm = p0.slice();
      pp[i] += eps; pm[i] -= eps;
      const num = (f(pp) - f(pm)) / (2 * eps);
      const ana = g(p0)[i];
      maxRel = Math.max(maxRel, Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana)));
    }
    add('梯度检验 maxRelErr', maxRel, 1e-6);

    // [2-5] 训练收敛
    const res = NN.train(X, y, { lr: 0.5, iters: 20000, seed: 42 });
    const F = NN.forward(res.params, X);
    let correct = 0, minMargin = 1;
    for (let s = 0; s < 4; s++) {
      const pred = F.A2[s] >= 0.5 ? 1 : 0;
      if (pred === y[s]) correct++;
      const pc = y[s] === 1 ? F.A2[s] : 1 - F.A2[s];
      minMargin = Math.min(minMargin, pc);
    }
    add('XOR 准确率', correct / 4, 1, 'ge');
    add('最终 BCE 损失', res.history[res.history.length - 1], 0.05);
    add('正确类最小边际', minMargin, 0.5, 'ge');
    const h = res.history;
    const avg = function (arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; };
    add('损失收敛（末500/初始）', avg(h.slice(-500)) / h[0], 0.1);
    let maxW = 0;
    for (const v of flatten(res.params)) maxW = Math.max(maxW, Math.abs(v));
    add('权重幅度上界', maxW, 20);

    return { name: 'nn-forge', layer: 'L4 深度学习·MLP', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return NN;
});
