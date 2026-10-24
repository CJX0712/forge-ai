/*!
 * gmm-forge.js — 晨星旗舰 AI · L2 无监督学习：高斯混合模型（EM）
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · EM 算法 —— Dempster, Laird & Rubin 1977（Jensen 不等式保证似然单调）
 *   · log-sum-exp 稳定技巧（防指数溢出的数值金标准）
 *   · 软分配（责任度）→ 期望步 / 最大化步
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 责任度归一    Σₖ rᵢₖ = 1（每行是概率分布）
 *   [2] 权重归一      Σₖ πₖ = 1
 *   [3] 似然单调不降  LL(t+1) ≥ LL(t) − tol（EM 定理的数值铁证）
 *   [4] 参数恢复      均值误差 ≤ 0.3（vs 真值 {−2, 2}）
 *   [5] 模型选择      混合 LL ≥ 单高斯 LL（K=2 优于 K=1）
 *   [6] 方差正定      σ² ≥ 1e-9
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.GMMForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const GM = {};

  GM.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  GM.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const LOG2PI = Math.log(2 * Math.PI);

  /* 一维高斯的对数密度 */
  function logGauss(x, mu, var_) {
    return -0.5 * (LOG2PI + Math.log(var_)) - (x - mu) * (x - mu) / (2 * var_);
  }
  /* log-sum-exp 稳定求值 */
  function logsumexp(arr) {
    let m = -Infinity;
    for (const v of arr) if (v > m) m = v;
    let s = 0;
    for (const v of arr) s += Math.exp(v - m);
    return m + Math.log(s);
  }

  /* EM 训练一维 GMM；返回 { weights, means, vars, llHistory, resp } */
  GM.fit = function (x, K, opts) {
    opts = opts || {};
    const iters = opts.iters || 300;
    const n = x.length;

    // 分位数初始化（确定性强于随机）
    const sorted = x.slice().sort(function (a, b) { return a - b; });
    const means = [], vars = [], weights = [];
    for (let k = 0; k < K; k++) {
      means.push(sorted[Math.floor((k + 0.5) * n / K)]);
      vars.push(1);
      weights.push(1 / K);
    }

    const llHist = [];
    let resp = null;
    for (let it = 0; it < iters; it++) {
      // E 步：责任度 + 本轮总对数似然
      let ll = 0;
      resp = new Array(n);
      for (let i = 0; i < n; i++) {
        const logs = [];
        for (let k = 0; k < K; k++) logs.push(Math.log(weights[k]) + logGauss(x[i], means[k], vars[k]));
        const lse = logsumexp(logs);
        ll += lse;
        const row = [];
        for (let k = 0; k < K; k++) row.push(Math.exp(logs[k] - lse));
        resp[i] = row;
      }
      llHist.push(ll);
      // M 步
      for (let k = 0; k < K; k++) {
        let Nk = 0, sx = 0;
        for (let i = 0; i < n; i++) { Nk += resp[i][k]; sx += resp[i][k] * x[i]; }
        weights[k] = Nk / n;
        means[k] = sx / Nk;
        let sv = 0;
        for (let i = 0; i < n; i++) { const t = x[i] - means[k]; sv += resp[i][k] * t * t; }
        vars[k] = Math.max(sv / Nk, 1e-9);
      }
    }
    // 末轮 LL（用更新后参数重算，与 EM 定理口径一致）
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const logs = [];
      for (let k = 0; k < K; k++) logs.push(Math.log(weights[k]) + logGauss(x[i], means[k], vars[k]));
      ll += logsumexp(logs);
    }
    llHist.push(ll);
    return { weights: weights, means: means, vars: vars, llHistory: llHist, resp: resp };
  };

  /* ================= 自检 ================= */
  GM.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = GM.mulberry32(42);
    const trueW = [0.6, 0.4], trueMu = [-2, 2], trueVar = [0.25, 0.25];
    const n = 200, x = [];
    for (let i = 0; i < n; i++) {
      const k = rng() < trueW[0] ? 0 : 1;
      x.push(trueMu[k] + Math.sqrt(trueVar[k]) * GM.randn(rng));
    }

    const res = GM.fit(x, 2, { iters: 300 });

    // [1] 责任度行和
    let rowErr = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < 2; k++) s += res.resp[i][k];
      rowErr = Math.max(rowErr, Math.abs(s - 1));
    }
    add('责任度行和 Σₖrᵢₖ=1', rowErr, 1e-12);

    // [2] 权重和
    add('权重和 Σπₖ=1', Math.abs(res.weights[0] + res.weights[1] - 1), 1e-12);

    // [3] 对数似然单调不降
    let maxDrop = 0;
    for (let i = 1; i < res.llHistory.length; i++) {
      maxDrop = Math.max(maxDrop, res.llHistory[i - 1] - res.llHistory[i]);
    }
    add('对数似然单调不降违例', maxDrop, 1e-6);

    // [4] 参数恢复
    const muSorted = res.means.slice().sort(function (a, b) { return a - b; });
    const muErr = Math.max(Math.abs(muSorted[0] - trueMu[0]), Math.abs(muSorted[1] - trueMu[1]));
    add('均值恢复误差', muErr, 0.3);

    // [5] 模型选择：混合 vs 单高斯
    let sMu = 0, sVar = 0;
    for (const v of x) sMu += v;
    sMu /= n;
    for (const v of x) sVar += (v - sMu) * (v - sMu);
    sVar /= n;
    let singleLL = 0;
    for (const v of x) singleLL += logGauss(v, sMu, sVar);
    const mixLL = res.llHistory[res.llHistory.length - 1];
    add('混合 LL − 单高斯 LL', mixLL - singleLL, 0, 'ge');

    // [6] 方差正定
    add('最小方差', Math.min.apply(null, res.vars), 1e-9, 'ge');

    return { name: 'gmm-forge', layer: 'L2 无监督·GMM/EM', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return GM;
});
