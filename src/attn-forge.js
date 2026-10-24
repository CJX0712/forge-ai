/*!
 * attn-forge.js — 晨星旗舰 AI · L5 注意力：缩放点积注意力 + 多头机制
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · 缩放点积注意力 —— Vaswani et al. 2017《Attention Is All You Need》
 *   · 多头并行 + 稳定 softmax（减行最大值防溢出）
 *   · softmax Jacobian 反向（dS = A ⊙ (dA − Σ(dA⊙A))）
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] softmax 行和    Σⱼ Aᵢⱼ = 1（每行是概率分布）
 *   [2] 权重有界        A ∈ [0,1]
 *   [3] 梯度检验        解析梯度 vs 中心差分 maxRelErr ≤ 1e-6（64 参数全量）
 *   [4] 上下文混合      改动 token 3 ⇒ token 0 输出变化 > 1e-3（注意力确实跨 token 交换信息）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AttnForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const AT = {};

  AT.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  AT.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  function softmax(row) {
    let m = -Infinity;
    for (const v of row) if (v > m) m = v;
    let s = 0;
    const e = row.map(function (v) { const t = Math.exp(v - m); s += t; return t; });
    return e.map(function (v) { return v / s; });
  }
  function matMul(A, B) {
    const r = A.length, k = B.length, c = B[0].length, C = [];
    for (let i = 0; i < r; i++) {
      C.push(new Array(c).fill(0));
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        for (let j = 0; j < c; j++) C[i][j] += a * B[p][j];
      }
    }
    return C;
  }
  function transpose(A) {
    const r = A.length, c = A[0].length, T = [];
    for (let j = 0; j < c; j++) { T.push([]); for (let i = 0; i < r; i++) T[j].push(A[i][j]); }
    return T;
  }
  function zeros(r, c) {
    const m = [];
    for (let i = 0; i < r; i++) m.push(new Array(c).fill(0));
    return m;
  }

  /* 多头前向：X(T×d), W{Wq,Wk,Wv,Wo}(d×d), heads h */
  AT.forward = function (X, W) {
    const T = X.length, d = X[0].length, h = W.heads, dk = d / h;
    const Q = matMul(X, W.Wq), K = matMul(X, W.Wk), V = matMul(X, W.Wv);
    const heads = [];
    for (let hi = 0; hi < h; hi++) {
      const cols = [];
      for (let j = 0; j < dk; j++) cols.push(hi * dk + j);
      const Qh = Q.map(function (r) { return cols.map(function (c) { return r[c]; }); });
      const Kh = K.map(function (r) { return cols.map(function (c) { return r[c]; }); });
      const Vh = V.map(function (r) { return cols.map(function (c) { return r[c]; }); });
      const S = [];
      for (let i = 0; i < T; i++) {
        const row = [];
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let k = 0; k < dk; k++) s += Qh[i][k] * Kh[j][k];
          row.push(s / Math.sqrt(dk));
        }
        S.push(row);
      }
      const A = S.map(softmax);
      const Oh = [];
      for (let i = 0; i < T; i++) {
        const row = new Array(dk).fill(0);
        for (let j = 0; j < T; j++) for (let k = 0; k < dk; k++) row[k] += A[i][j] * Vh[j][k];
        Oh.push(row);
      }
      heads.push({ cols: cols, Qh: Qh, Kh: Kh, Vh: Vh, S: S, A: A, Oh: Oh });
    }
    const O = [];
    for (let i = 0; i < T; i++) {
      const row = new Array(d).fill(0);
      for (let hi = 0; hi < h; hi++) for (let k = 0; k < dk; k++) row[hi * dk + k] = heads[hi].Oh[i][k];
      O.push(row);
    }
    const out = matMul(O, W.Wo);
    return { out: out, O: O, Q: Q, K: K, V: V, heads: heads, T: T, d: d, X: X };
  }

  /* 反向：给定 dout(T×d)，返回各权重梯度 */
  AT.backward = function (dout, F, W) {
    const T = F.T, d = F.d, h = W.heads, dk = d / h;
    const dWo = matMul(transpose(F.O), dout);
    const dO = matMul(dout, transpose(W.Wo));
    const dQ = zeros(T, d), dK = zeros(T, d), dV = zeros(T, d);
    for (let hi = 0; hi < h; hi++) {
      const hd = F.heads[hi], cols = hd.cols;
      const dOh = dO.map(function (r) { return cols.map(function (c) { return r[c]; }); });
      const dVh = matMul(transpose(hd.A), dOh);
      const dAh = matMul(dOh, transpose(hd.Vh));
      const dSh = [];
      for (let i = 0; i < T; i++) {
        const row = [];
        let dot = 0;
        for (let j = 0; j < T; j++) dot += dAh[i][j] * hd.A[i][j];
        for (let j = 0; j < T; j++) row.push(hd.A[i][j] * (dAh[i][j] - dot));
        dSh.push(row);
      }
      const dQh = matMul(dSh, hd.Kh).map(function (r) { return r.map(function (v) { return v / Math.sqrt(dk); }); });
      const dKh = matMul(transpose(dSh), hd.Qh).map(function (r) { return r.map(function (v) { return v / Math.sqrt(dk); }); });
      for (let i = 0; i < T; i++) for (let k = 0; k < dk; k++) {
        dQ[i][cols[k]] = dQh[i][k];
        dK[i][cols[k]] = dKh[i][k];
        dV[i][cols[k]] = dVh[i][k];
      }
    }
    return {
      dWq: matMul(transpose(F.X), dQ),
      dWk: matMul(transpose(F.X), dK),
      dWv: matMul(transpose(F.X), dV),
      dWo: dWo
    };
  };

  /* ================= 自检 ================= */
  AT.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = AT.mulberry32(20261024);
    const T = 5, d = 4, h = 2;
    const X = [];
    for (let i = 0; i < T; i++) { X.push([]); for (let j = 0; j < d; j++) X[i].push(AT.randn(rng)); }
    const W = { heads: h, Wq: [], Wk: [], Wv: [], Wo: [] };
    for (const key of ['Wq', 'Wk', 'Wv', 'Wo']) {
      for (let i = 0; i < d; i++) { W[key].push([]); for (let j = 0; j < d; j++) W[key][i].push(AT.randn(rng) * 0.5); }
    }

    const F = AT.forward(X, W);

    // [1] softmax 行和
    let rowErr = 0;
    for (const hd of F.heads) for (const row of hd.A) {
      let s = 0;
      for (const v of row) s += v;
      rowErr = Math.max(rowErr, Math.abs(s - 1));
    }
    add('softmax 行和 ΣⱼAᵢⱼ=1', rowErr, 1e-12);

    // [2] 权重有界
    let rangeViol = 0;
    for (const hd of F.heads) for (const row of hd.A) for (const v of row) {
      rangeViol = Math.max(rangeViol, -v, v - 1);
    }
    add('注意力权重越界量', rangeViol, 0);

    // [3] 梯度检验：f = 0.5·Σout²，参数 = [Wq|Wk|Wv|Wo] 共 64 维
    const keys = ['Wq', 'Wk', 'Wv', 'Wo'];
    const f = function (flat) {
      const Wc = { heads: h };
      let off = 0;
      for (const key of keys) {
        Wc[key] = [];
        for (let i = 0; i < d; i++) { Wc[key].push(flat.slice(off, off + d)); off += d; }
      }
      const Fc = AT.forward(X, Wc);
      let s = 0;
      for (const row of Fc.out) for (const v of row) s += v * v;
      return 0.5 * s;
    };
    const g = function (flat) {
      const Wc = { heads: h };
      let off = 0;
      for (const key of keys) {
        Wc[key] = [];
        for (let i = 0; i < d; i++) { Wc[key].push(flat.slice(off, off + d)); off += d; }
      }
      const Fc = AT.forward(X, Wc);
      const dout = Fc.out; // d(0.5Σout²)/dout = out
      const G = AT.backward(dout, Fc, Wc);
      return [].concat.apply([], keys.map(function (key) { return [].concat.apply([], G['d' + key]); }));
    };
    const flat0 = [].concat.apply([], keys.map(function (key) { return [].concat.apply([], W[key]); }));
    let maxRel = 0;
    for (let i = 0; i < flat0.length; i++) {
      const eps = 1e-5;
      const pp = flat0.slice(), pm = flat0.slice();
      pp[i] += eps; pm[i] -= eps;
      const num = (f(pp) - f(pm)) / (2 * eps);
      const ana = g(flat0)[i];
      maxRel = Math.max(maxRel, Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana)));
    }
    add('梯度检验 maxRelErr（64 参数）', maxRel, 1e-6);

    // [4] 上下文混合：扰动 token 3 ⇒ token 0 输出变化
    const Xp = X.map(function (row, i) { return i === 3 ? row.map(function (v) { return v + 0.7; }) : row.slice(); });
    const Fp = AT.forward(Xp, W);
    let delta = 0;
    for (let j = 0; j < d; j++) delta = Math.max(delta, Math.abs(Fp.out[0][j] - F.out[0][j]));
    add('上下文混合 Δout(token0 | Δtoken3)', delta, 1e-3, 'ge');

    return { name: 'attn-forge', layer: 'L5 注意力·Transformer', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return AT;
});
