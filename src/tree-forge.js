/*!
 * tree-forge.js — 晨星旗舰 AI · L1 监督学习：CART 决策树
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · CART 递归二分 + Gini 不纯度 —— Breiman, Friedman, Olshen, Stone 1984
 *   · 预剪枝（maxDepth / minSplit）抑制过拟合
 *   · 连续特征排序扫描，O(N·d·logN) 建树
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 训练准确率     = 1（可分开数据完全拟合）
 *   [2] 分裂增益       ≥ 0（每个内部节点 Gini 严格不增）
 *   [3] 叶子纯度       Gini = 0（纯叶）
 *   [4] 深度不变量     depth ≤ maxDepth
 *   [5] 父-子不纯度    父 Gini ≥ 加权子 Gini（分裂定义本身）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TreeForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const TR = {};

  TR.gini = function (labels) {
    const n = labels.length;
    if (n === 0) return 0;
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    let s = 0;
    for (const k in counts) { const p = counts[k] / n; s += p * p; }
    return 1 - s;
  };

  /* 全特征扫描最优二分：返回 {feature, threshold, gain} */
  TR.bestSplit = function (X, y) {
    const n = X.length, d = X[0].length;
    let best = { gain: -1, feature: -1, threshold: 0 };
    for (let f = 0; f < d; f++) {
      const idx = [];
      for (let i = 0; i < n; i++) idx.push(i);
      idx.sort(function (a, b) { return X[a][f] - X[b][f]; });
      const left = {}, right = {};
      for (const i of idx) right[y[i]] = (right[y[i]] || 0) + 1;
      let nl = 0;
      for (let s = 0; s < n - 1; s++) {
        const i = idx[s];
        left[y[i]] = (left[y[i]] || 0) + 1;
        right[y[i]]--;
        nl = s + 1;
        if (X[idx[s + 1]][f] === X[i][f]) continue; // 仅在取值变化处分裂
        const nr = n - nl;
        let gl = 1, gr = 1;
        for (const k in left) { const p = left[k] / nl; gl -= p * p; }
        for (const k in right) { const p = right[k] / nr; gr -= p * p; }
        const gain = TR.gini(y) - (nl / n) * gl - (nr / n) * gr;
        if (gain > best.gain + 1e-15) {
          best = { gain: gain, feature: f, threshold: (X[i][f] + X[idx[s + 1]][f]) / 2 };
        }
      }
    }
    return best;
  };

  TR.grow = function (X, y, opts) {
    opts = opts || {};
    const maxDepth = opts.maxDepth != null ? opts.maxDepth : 8;
    const minSplit = opts.minSplit || 2;
    const depth = opts.depth || 0;
    const g = TR.gini(y);
    const node = { depth: depth, gini: g, n: X.length };
    const majority = (function () {
      const c = {};
      for (const l of y) c[l] = (c[l] || 0) + 1;
      let bestL = y[0], bestC = -1;
      for (const k in c) if (c[k] > bestC) { bestC = c[k]; bestL = +k; }
      return bestL;
    })();
    if (g <= 1e-12 || depth >= maxDepth || X.length < minSplit) {
      node.leaf = true; node.label = majority;
      return node;
    }
    const bs = TR.bestSplit(X, y);
    if (bs.gain <= 1e-12) {
      node.leaf = true; node.label = majority;
      return node;
    }
    node.leaf = false;
    node.feature = bs.feature;
    node.threshold = bs.threshold;
    node.gain = bs.gain;
    const li = [], ri = [];
    for (let i = 0; i < X.length; i++) (X[i][bs.feature] <= bs.threshold ? li : ri).push(i);
    node.left = TR.grow(li.map(function (i) { return X[i]; }), li.map(function (i) { return y[i]; }),
      { maxDepth: maxDepth, minSplit: minSplit, depth: depth + 1 });
    node.right = TR.grow(ri.map(function (i) { return X[i]; }), ri.map(function (i) { return y[i]; }),
      { maxDepth: maxDepth, minSplit: minSplit, depth: depth + 1 });
    return node;
  };

  TR.predictOne = function (node, x) {
    while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
    return node.label;
  };

  function walk(node, fn) {
    fn(node);
    if (!node.leaf) { walk(node.left, fn); walk(node.right, fn); }
  }

  /* ================= 自检 ================= */
  TR.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    // 两类高斯团 + 清晰间隔（可完全分开）
    const DATA = [
      [1, 1, 0], [1, 2, 0], [2, 1, 0], [2, 2, 0], [1.5, 1.5, 0], [2.5, 2.5, 0],
      [6, 6, 1], [7, 6, 1], [6, 7, 1], [7, 7, 1], [6.5, 6.5, 1], [7.5, 7.5, 1]
    ];
    const X = DATA.map(function (r) { return [r[0], r[1]]; });
    const y = DATA.map(function (r) { return r[2]; });

    const root = TR.grow(X, y, { maxDepth: 4, minSplit: 2 });

    let correct = 0;
    for (let i = 0; i < X.length; i++) if (TR.predictOne(root, X[i]) === y[i]) correct++;
    add('训练准确率', correct / X.length, 1, 'ge');

    let minGain = Infinity, maxDepth = 0, maxLeafGini = 0, minImpurityDrop = Infinity;
    walk(root, function (nd) {
      maxDepth = Math.max(maxDepth, nd.depth);
      if (nd.leaf) maxLeafGini = Math.max(maxLeafGini, nd.gini);
      else {
        minGain = Math.min(minGain, nd.gain);
        const wl = nd.left.n / nd.n, wr = nd.right.n / nd.n;
        minImpurityDrop = Math.min(minImpurityDrop, nd.gini - (wl * nd.left.gini + wr * nd.right.gini));
      }
    });
    add('分裂增益最小值', minGain, 0, 'ge');
    add('叶子 Gini 最大值', maxLeafGini, 1e-12);
    add('树深度 ≤ maxDepth', maxDepth, 4);
    add('父-子不纯度下降最小值', minImpurityDrop, 0, 'ge');

    return { name: 'tree-forge', layer: 'L1 监督·决策树 CART', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return TR;
});
