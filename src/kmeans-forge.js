/*!
 * kmeans-forge.js — 晨星旗舰 AI · L2 无监督学习：K-Means 聚类
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · Lloyd 迭代（Lloyd 1982；MacQueen 1967 在线版）
 *   · 分配-更新两步法，目标函数 SSE 单调不增（收敛性定理）
 *   · 空簇保护（保留原质心，SSE 不变 ⇒ 不破坏单调性）
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] SSE 单调不增   每轮迭代 ΔSSE ≤ 1e-9（Lloyd 收敛定理的数值体现）
 *   [2] 目标函数下降   最终 SSE / 总散布 ≤ 0.15
 *   [3] 质心合法性     质心 ⊆ 数据包围盒
 *   [4] 簇非空         min|cluster| ≥ 1
 *   [5] 分配一致性     每点确属最近质心（欧氏）
 *   [6] 聚类纯度       ≥ 0.95（vs 真实标签）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KMeansForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const KM = {};

  KM.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  KM.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  KM.fit = function (X, k, opts) {
    opts = opts || {};
    const iters = opts.iters || 100;
    const rng = opts.rng || KM.mulberry32(42);
    const n = X.length, d = X[0].length;

    // k-means++ 风格的确定性初始化：取 k 个互异样本点
    const centroids = [];
    const used = {};
    while (centroids.length < k) {
      const i = Math.floor(rng() * n);
      if (used[i]) continue;
      used[i] = 1;
      centroids.push(X[i].slice());
    }

    let assign = new Array(n).fill(0);
    let sse = Infinity;
    const sseHist = [];
    for (let it = 0; it < iters; it++) {
      // 分配步：最近质心
      let cur = 0;
      for (let i = 0; i < n; i++) {
        let best = 0, bestD = Infinity;
        for (let c = 0; c < k; c++) {
          let dd = 0;
          for (let j = 0; j < d; j++) { const t = X[i][j] - centroids[c][j]; dd += t * t; }
          if (dd < bestD) { bestD = dd; best = c; }
        }
        assign[i] = best;
        cur += bestD;
      }
      sse = cur;
      sseHist.push(sse);
      // 更新步：均值；空簇保留原质心
      const sums = [];
      for (let c = 0; c < k; c++) sums.push({ s: new Array(d).fill(0), n: 0 });
      for (let i = 0; i < n; i++) {
        const a = assign[i];
        for (let j = 0; j < d; j++) sums[a].s[j] += X[i][j];
        sums[a].n++;
      }
      for (let c = 0; c < k; c++) {
        if (sums[c].n === 0) continue;
        for (let j = 0; j < d; j++) centroids[c][j] = sums[c].s[j] / sums[c].n;
      }
    }
    return { centroids: centroids, assign: assign, sse: sse, sseHistory: sseHist };
  };

  /* ================= 自检 ================= */
  KM.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = KM.mulberry32(42);
    const centers = [[0, 0], [6, 0], [0, 6]];
    const X = [], truth = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 30; i++) {
        X.push([centers[c][0] + 0.5 * KM.randn(rng), centers[c][1] + 0.5 * KM.randn(rng)]);
        truth.push(c);
      }
    }
    const n = X.length;

    const res = KM.fit(X, 3, { iters: 100, rng: rng });

    // [1] SSE 单调不增
    let maxRise = 0;
    for (let i = 1; i < res.sseHistory.length; i++) {
      maxRise = Math.max(maxRise, res.sseHistory[i] - res.sseHistory[i - 1]);
    }
    add('SSE 单调不增违例', maxRise, 1e-9);

    // [2] 目标函数下降
    let totalScatter = 0;
    const gmean = [0, 0];
    for (const row of X) { gmean[0] += row[0] / n; gmean[1] += row[1] / n; }
    for (const row of X) {
      totalScatter += (row[0] - gmean[0]) * (row[0] - gmean[0]) + (row[1] - gmean[1]) * (row[1] - gmean[1]);
    }
    add('最终 SSE / 总散布', res.sse / totalScatter, 0.15);

    // [3] 质心在数据包围盒内
    let bbViol = 0;
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const row of X) for (let j = 0; j < 2; j++) { lo[j] = Math.min(lo[j], row[j]); hi[j] = Math.max(hi[j], row[j]); }
    for (const c of res.centroids) for (let j = 0; j < 2; j++) {
      bbViol = Math.max(bbViol, lo[j] - c[j], c[j] - hi[j]);
    }
    add('质心越界量', bbViol, 1e-12);

    // [4] 簇非空
    const sizes = [0, 0, 0];
    for (const a of res.assign) sizes[a]++;
    add('最小簇规模', Math.min.apply(null, sizes), 1, 'ge');

    // [5] 分配一致性
    let asgErr = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < 3; c++) {
        let dd = 0;
        for (let j = 0; j < 2; j++) { const t = X[i][j] - res.centroids[c][j]; dd += t * t; }
        if (dd < bestD) { bestD = dd; best = c; }
      }
      if (best !== res.assign[i]) asgErr++;
    }
    add('分配不一致点数', asgErr, 0);

    // [6] 聚类纯度（多数投票）
    let pure = 0;
    for (let c = 0; c < 3; c++) {
      const cnt = {};
      for (let i = 0; i < n; i++) if (res.assign[i] === c) cnt[truth[i]] = (cnt[truth[i]] || 0) + 1;
      let mx = 0;
      for (const kk in cnt) mx = Math.max(mx, cnt[kk]);
      pure += mx;
    }
    add('聚类纯度', pure / n, 0.95, 'ge');

    return { name: 'kmeans-forge', layer: 'L2 无监督·K-Means', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return KM;
});
