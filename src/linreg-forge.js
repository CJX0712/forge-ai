/*!
 * linreg-forge.js — 晨星旗舰 AI · L1 监督学习：线性回归（Ridge）
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · 正规方程闭式解  —— Gauss–Markov 定理：OLS 是最优线性无偏估计（BLUE）
 *   · 批量梯度下降    —— Cauchy 1847；凸性 ⇒ 全局最优，两路独立收敛互证
 *   · Ridge 正则      —— Hoerl & Kennard 1970：多重共线性 → 病态的正交化解药
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 正规方程残差   Xᵀ(y − Xŵ) = 0（OLS 一阶条件，金标准）
 *   [2] 闭式解 vs GD  两条独立优化路径收敛到同一最优点
 *   [3] R² ≥ 0.9      信号可恢复
 *   [4] 权重恢复       |ŵ − w_true|∞ 有界（噪声 0.1 下的统计一致性）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LinRegForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const LR = {};

  LR.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  LR.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  function transpose(A) {
    const r = A.length, c = A[0].length, T = [];
    for (let j = 0; j < c; j++) { T.push([]); for (let i = 0; i < r; i++) T[j].push(A[i][j]); }
    return T;
  }
  function matVec(A, x) {
    const y = new Array(A.length).fill(0);
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      for (let j = 0; j < x.length; j++) s += A[i][j] * x[j];
      y[i] = s;
    }
    return y;
  }
  function cholesky(A) {
    const n = A.length, L = [];
    for (let i = 0; i < n; i++) { L.push(new Array(n).fill(0)); }
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
    return L;
  }
  function choleskySolve(L, b) {
    const n = L.length, y = new Array(n).fill(0), x = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  /* ---------- 闭式解：(XᵀX + λI)⁻¹Xᵀy，Cholesky 直接求解 ---------- */
  LR.fitClosedForm = function (X, y, lambda) {
    const d = X[0].length;
    const Xt = transpose(X);
    const XtX = [];
    for (let i = 0; i < d; i++) {
      XtX.push([]);
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let k = 0; k < X.length; k++) s += Xt[i][k] * X[k][j];
        XtX[i].push(s);
      }
    }
    for (let i = 0; i < d; i++) XtX[i][i] += (lambda || 0);
    const Xty = matVec(Xt, y);
    return choleskySolve(cholesky(XtX), Xty);
  };

  /* ---------- 批量梯度下降（与闭式解同一目标函数） ---------- */
  LR.fitGradientDescent = function (X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.05, iters = opts.iters || 20000, lambda = opts.lambda || 0;
    const n = X.length, d = X[0].length;
    const w = new Array(d).fill(0);
    const Xt = transpose(X);
    const hist = [];
    for (let it = 0; it < iters; it++) {
      const pred = matVec(X, w);
      const err = pred.map(function (p, i) { return p - y[i]; });
      const g = matVec(Xt, err);
      for (let j = 0; j < d; j++) w[j] -= lr * (g[j] / n + lambda * w[j]);
      let s = 0;
      for (let i = 0; i < n; i++) s += err[i] * err[i];
      let r2 = 0;
      for (let j = 0; j < d; j++) r2 += w[j] * w[j];
      hist.push(s / (2 * n) + lambda / 2 * r2);
    }
    return { w: w, history: hist };
  };

  LR.predict = function (X, w) { return matVec(X, w); };

  /* ================= 自检 ================= */
  LR.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = LR.mulberry32(20261024);
    const n = 100;
    const wTrue = [3, 2, -1, 0.5]; // [bias, x1, x2, x3]
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const row = [1, LR.randn(rng), LR.randn(rng), LR.randn(rng)];
      X.push(row);
      y.push(wTrue[0] + wTrue[1] * row[1] + wTrue[2] * row[2] + wTrue[3] * row[3] + 0.1 * LR.randn(rng));
    }

    const lam = 1e-8;
    const wCf = LR.fitClosedForm(X, y, lam);
    const gd = LR.fitGradientDescent(X, y, { lr: 0.05, iters: 20000, lambda: lam });
    const wGd = gd.w;

    let diff = 0;
    for (let j = 0; j < 4; j++) diff = Math.max(diff, Math.abs(wCf[j] - wGd[j]));
    add('闭式解 vs 梯度下降一致', diff, 0.01);

    const Xt = transpose(X);
    const pred = LR.predict(X, wCf);
    const r = y.map(function (v, i) { return v - pred[i]; });
    const Xtr = matVec(Xt, r), Xty = matVec(Xt, y);
    let num = 0, den = 0;
    for (let j = 0; j < 4; j++) { num += Xtr[j] * Xtr[j]; den += Xty[j] * Xty[j]; }
    add('正规方程 ‖Xᵀ(y−Xŵ)‖/‖Xᵀy‖', Math.sqrt(num / den), 1e-10);

    const ybar = y.reduce(function (a, b) { return a + b; }, 0) / n;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < n; i++) { ssRes += (y[i] - pred[i]) * (y[i] - pred[i]); ssTot += (y[i] - ybar) * (y[i] - ybar); }
    add('R² 决定系数', 1 - ssRes / ssTot, 0.9, 'ge');

    let wErr = 0;
    for (let j = 0; j < 4; j++) wErr = Math.max(wErr, Math.abs(wCf[j] - wTrue[j]));
    add('权重恢复 |ŵ−w_true|∞', wErr, 0.25);

    const h = gd.history;
    const avg = function (arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; };
    add('GD 损失收敛（末500/首500）', avg(h.slice(-500)) / avg(h.slice(0, 500)), 0.05);

    return { name: 'linreg-forge', layer: 'L1 监督·线性回归', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return LR;
});
