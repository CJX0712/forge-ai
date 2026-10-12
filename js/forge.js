/**
 * forge.js —— forge-ai 单文件 JS 内核（零依赖 · 数学不变量可验证）
 *
 * 纯 JS 从零实现：Cholesky / Jacobi 特征分解 / 单侧 Jacobi SVD / PCA / GMM(EM)
 * / MLP 反向传播 / 缩放点积注意力 / kNN。
 *
 * 运行自检：  node forge.js          （headless，8 项全绿则 exit 0）
 * 浏览器：    引入本文件后调用 ForgeAI.selfTest()
 *
 * 作者：晨星
 */
'use strict';

// ============================ 基础工具 ============================
function zeros(r, c) {
  const a = new Array(r);
  for (let i = 0; i < r; i++) a[i] = new Float64Array(c);
  return a;
}
function eye(n) {
  const a = zeros(n, n);
  for (let i = 0; i < n; i++) a[i][i] = 1;
  return a;
}
function clone(A) { return A.map(r => r.slice()); }
function col(A, j) { const v = new Float64Array(A.length); for (let i = 0; i < A.length; i++) v[i] = A[i][j]; return v; }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function transpose(A) {
  const r = A.length, c = A[0].length, T = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
  return T;
}
function matmul(A, B) {
  const n = A.length, p = B.length, m = B[0].length, C = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let k = 0; k < p; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j];
    }
  return C;
}
function maxAbs(A) { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) m = Math.max(m, Math.abs(A[i][j])); return m; }

// 可复现随机（mulberry32 + Box–Muller）
function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { u, n: () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u()) };
}

// ============================ 分解 ============================
function cholesky(A) {
  const n = A.length, L = zeros(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(A[i][i] - s, 1e-12));
      else L[i][j] = (A[i][j] - s) / L[j][j];
    }
  return L;
}

function jacobiEig(A, maxSweeps = 200, tol = 1e-12) {
  const n = A.length;
  const a = A.map(r => Array.from(r));
  const V = eye(n).map(r => Array.from(r));
  for (let sw = 0; sw < maxSweeps; sw++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        let t;
        if (Math.abs(a[p][p] - a[q][q]) < 1e-300) t = a[p][q] >= 0 ? 1 : -1;
        else {
          const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          t = Math.sign(th) / (Math.abs(th) + Math.sqrt(th * th + 1));
        }
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        const App = a[p][p], Aqq = a[q][q], Apq = a[p][q];
        a[p][p] = c * c * App - 2 * s * c * Apq + s * s * Aqq;
        a[q][q] = s * s * App + 2 * s * c * Apq + c * c * Aqq;
        a[p][q] = 0; a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const Akp = a[k][p], Akq = a[k][q];
          a[k][p] = c * Akp - s * Akq; a[k][q] = s * Akp + c * Akq;
          a[p][k] = a[k][p]; a[q][k] = a[k][q];
        }
        for (let k = 0; k < n; k++) {
          const Vkp = V[k][p], Vkq = V[k][q];
          V[k][p] = c * Vkp - s * Vkq; V[k][q] = s * Vkp + c * Vkq;
        }
      }
  }
  const vals = [];
  for (let i = 0; i < n; i++) vals.push(a[i][i]);
  const idx = vals.map((v, i) => i).sort((x, y) => vals[x] - vals[y]);
  return { values: idx.map(i => vals[i]), vectors: idx.map(i => col(V, i)) };
}

// 单侧 Jacobi SVD（Hestenes 1960）
function svd(A) {
  const m = A.length, n = A[0].length;
  const a = A.map(r => Array.from(r));
  let V = eye(n).map(r => Array.from(r));
  for (let sw = 0; sw < 50; sw++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += Math.pow(dot(col(a, p), col(a, q)), 2);
    if (off < 1e-12) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        const ap = col(a, p), aq = col(a, q);
        const g = dot(ap, aq);
        if (Math.abs(g) < 1e-300) continue;
        const al = dot(ap, ap), be = dot(aq, aq);
        const z = (be - al) / (2 * g);
        const t = Math.sign(z) / (Math.abs(z) + Math.sqrt(z * z + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let i = 0; i < m; i++) { const x = a[i][p], y = a[i][q]; a[i][p] = c * x - s * y; a[i][q] = s * x + c * y; }
        for (let i = 0; i < n; i++) { const x = V[i][p], y = V[i][q]; V[i][p] = c * x - s * y; V[i][q] = s * x + c * y; }
      }
  }
  const sig = [];
  for (let j = 0; j < n; j++) sig.push(Math.sqrt(dot(col(a, j), col(a, j))));
  const idx = sig.map((v, i) => i).sort((x, y) => sig[y] - sig[x]);
  const k = Math.min(m, n);
  const U = zeros(m, k), Vt = zeros(k, n), s = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    const oi = idx[j];
    const sv = Math.max(sig[oi], 1e-12);
    for (let i = 0; i < m; i++) U[i][j] = a[i][oi] / sv;
    s[j] = sig[oi];
    const vc = col(V, oi);
    for (let i = 0; i < n; i++) Vt[j][i] = vc[i];
  }
  return { U, s, Vt };
}

// ============================ PCA ============================
function pcaFit(X, nComp) {
  const n = X.length, d = X[0].length;
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j] / n;
  const Xc = zeros(n, d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Xc[i][j] = X[i][j] - mean[j];
  const C = zeros(d, d);
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += Xc[i][a] * Xc[i][b] / (n - 1);
  const { values, vectors } = jacobiEig(C);
  const order = values.map((v, i) => i).sort((x, y) => values[y] - values[x]);
  const k = nComp || d;
  const comps = zeros(k, d);
  for (let j = 0; j < k; j++) { const vc = vectors[order[j]]; for (let i = 0; i < d; i++) comps[j][i] = vc[i]; }
  return { mean, components: comps, explainedVariance: order.slice(0, k).map(i => values[i]) };
}
function pcaTransform(model, X) {
  const n = X.length, d = X[0].length, k = model.components.length;
  const Z = zeros(n, k);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) { let s = 0; for (let t = 0; t < d; t++) s += (X[i][t] - model.mean[t]) * model.components[j][t]; Z[i][j] = s; }
  return Z;
}
function pcaInverse(model, Z) {
  const n = Z.length, k = model.components.length, d = model.components[0].length;
  const X = zeros(n, d);
  for (let i = 0; i < n; i++)
    for (let t = 0; t < d; t++) { let s = model.mean[t]; for (let j = 0; j < k; j++) s += Z[i][j] * model.components[j][t]; X[i][t] = s; }
  return X;
}

// ============================ GMM (EM) ============================
function logsumexp(row) {
  let mx = -Infinity;
  for (const v of row) mx = Math.max(mx, v);
  let s = 0;
  for (const v of row) s += Math.exp(v - mx);
  return mx + Math.log(s);
}
function gmmFit(X, K, opts = {}) {
  const nIter = opts.nIter || 60, seed = opts.seed || 0;
  const n = X.length, d = X[0].length, r = rng(seed);
  // 随机初始化中心（简单稳健）
  const means = zeros(K, d);
  for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) means[k][j] = X[Math.floor(r.u() * n)][j];
  let weights = new Float64Array(K).fill(1 / K);
  // 全局协方差初始化
  let cov0 = zeros(d, d);
  for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov0[a][b] += X[i][a] * X[i][b] / n;
  for (let a = 0; a < d; a++) cov0[a][a] += 1e-6;
  let covs = []; for (let k = 0; k < K; k++) covs.push(clone(cov0));
  const hist = [];
  let resp = zeros(n, K);
  for (let it = 0; it < nIter; it++) {
    const logResp = zeros(n, K);
    for (let k = 0; k < K; k++) {
      const Lc = cholesky(covs[k]);
      let logdet = 0;
      for (let a = 0; a < d; a++) logdet += 2 * Math.log(Lc[a][a]);
      for (let i = 0; i < n; i++) {
        // 解 L y = (x - mu)
        const y = new Float64Array(d);
        for (let a = 0; a < d; a++) { let s = X[i][a] - means[k][a]; for (let b = 0; b < a; b++) s -= Lc[a][b] * y[b]; y[a] = s / Lc[a][a]; }
        let quad = 0; for (let a = 0; a < d; a++) quad += y[a] * y[a];
        logResp[i][k] = -0.5 * (d * Math.log(2 * Math.PI) + logdet + quad) + Math.log(weights[k]);
      }
    }
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const row = []; for (let k = 0; k < K; k++) row.push(logResp[i][k]);
      const ls = logsumexp(row);
      ll += ls;
      for (let k = 0; k < K; k++) resp[i][k] = Math.exp(logResp[i][k] - ls);
    }
    hist.push(ll);
    // M-step
    const Nk = new Float64Array(K);
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += resp[i][k];
    for (let k = 0; k < K; k++) weights[k] = (Nk[k] + 1e-10) / n;
    for (let k = 0; k < K; k++)
      for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += resp[i][k] * X[i][j]; means[k][j] = s / (Nk[k] + 1e-10); }
    covs = [];
    for (let k = 0; k < K; k++) {
      const C = zeros(d, d);
      for (let i = 0; i < n; i++) {
        const w = resp[i][k];
        const diff = new Float64Array(d);
        for (let a = 0; a < d; a++) diff[a] = X[i][a] - means[k][a];
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += w * diff[a] * diff[b];
      }
      for (let a = 0; a < d; a++) { for (let b = 0; b < d; b++) C[a][b] /= (Nk[k] + 1e-10); C[a][a] += 1e-6; }
      covs.push(C);
    }
  }
  return { weights, means, covs, resp, loglik: hist };
}

// ============================ MLP（含反向传播） ============================
function softmaxRows(Z) {
  const n = Z.length, m = Z[0].length, P = zeros(n, m);
  for (let i = 0; i < n; i++) {
    let mx = -Infinity; for (let j = 0; j < m; j++) mx = Math.max(mx, Z[i][j]);
    let s = 0; for (let j = 0; j < m; j++) { P[i][j] = Math.exp(Z[i][j] - mx); s += P[i][j]; }
    for (let j = 0; j < m; j++) P[i][j] /= s;
  }
  return P;
}
function mlpForward(W1, b1, W2, b2, X) {
  const Z1 = zeros(X.length, W1[0].length);
  const A1 = zeros(X.length, W1[0].length);
  for (let i = 0; i < X.length; i++)
    for (let j = 0; j < W1[0].length; j++) {
      let s = b1[j]; for (let t = 0; t < W1.length; t++) s += X[i][t] * W1[t][j];
      Z1[i][j] = s; A1[i][j] = Math.tanh(s);
    }
  const Z2 = matmul(A1, W2);
  for (let i = 0; i < Z2.length; i++) for (let j = 0; j < Z2[0].length; j++) Z2[i][j] += b2[j];
  return { P: softmaxRows(Z2), A1, Z2 };
}
function mlpLoss(W1, b1, W2, b2, X, y) {
  const { P } = mlpForward(W1, b1, W2, b2, X);
  let s = 0;
  for (let i = 0; i < X.length; i++) s -= Math.log(Math.max(P[i][y[i]], 1e-12));
  return s / X.length;
}
function mlpBackward(W1, b1, W2, b2, X, y) {
  const n = X.length;
  const { P, A1 } = mlpForward(W1, b1, W2, b2, X);
  const m = P[0].length;
  const dZ2 = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) dZ2[i][j] = (P[i][j] - (j === y[i] ? 1 : 0)) / n;
  const gW2 = matmul(transpose(A1), dZ2);
  const gb2 = new Float64Array(m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) gb2[j] += dZ2[i][j];
  const dA1 = matmul(dZ2, transpose(W2));
  const dZ1 = zeros(n, A1[0].length);
  for (let i = 0; i < n; i++) for (let j = 0; j < A1[0].length; j++) dZ1[i][j] = dA1[i][j] * (1 - A1[i][j] * A1[i][j]);
  const gW1 = matmul(transpose(X), dZ1);
  const gb1 = new Float64Array(A1[0].length);
  for (let i = 0; i < n; i++) for (let j = 0; j < A1[0].length; j++) gb1[j] += dZ1[i][j];
  return { gW1, gb1, gW2, gb2 };
}

// ============================ 缩放点积注意力（含反向） ============================
function attention(Q, K, V) {
  const n = Q.length, d = Q[0].length, kd = K[0].length;
  const scores = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) scores[i][j] = dot(Q[i], K[j]) / Math.sqrt(d);
  const A = softmaxRows(scores);
  const out = zeros(n, kd);
  for (let i = 0; i < n; i++) for (let j = 0; j < kd; j++) { let s = 0; for (let t = 0; t < n; t++) s += A[i][t] * V[t][j]; out[i][j] = s; }
  return { out, A, Q, K, V };
}
function attentionBackward(dOut, cache) {
  const { A, Q, K, V } = cache;
  const n = Q.length, d = Q[0].length, kd = V[0].length;
  const dV = zeros(n, kd), dA = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < kd; j++) { let s = 0; for (let t = 0; t < n; t++) s += A[t][i] * dOut[t][j]; dV[i][j] = s; }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let t = 0; t < kd; t++) s += dOut[i][t] * V[j][t]; dA[i][j] = s; }
  const dScores = zeros(n, n);
  for (let i = 0; i < n; i++) {
    let row = 0; for (let j = 0; j < n; j++) row += dA[i][j] * A[i][j];
    for (let j = 0; j < n; j++) dScores[i][j] = A[i][j] * (dA[i][j] - row);
  }
  const dQ = zeros(n, d), dK = zeros(n, d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { let s = 0; for (let t = 0; t < n; t++) s += dScores[i][t] * K[t][j]; dQ[i][j] = s / Math.sqrt(d); }
  for (let j = 0; j < n; j++) for (let i = 0; i < d; i++) { let s = 0; for (let t = 0; t < n; t++) s += dScores[t][j] * Q[t][i]; dK[j][i] = s / Math.sqrt(d); }
  return { dQ, dK, dV };
}

// ============================ kNN ============================
function knnPredict(Xtr, ytr, X, k) {
  return X.map(x => {
    const dist = Xtr.map((xr, i) => {
      let s = 0; for (let j = 0; j < x.length; j++) { const d = xr[j] - x[j]; s += d * d; }
      return [Math.sqrt(s), ytr[i]];
    }).sort((a, b) => a[0] - b[0]);
    const votes = new Map();
    for (let i = 0; i < k; i++) votes.set(dist[i][1], (votes.get(dist[i][1]) || 0) + 1);
    let best = null, bc = -1;
    for (const [lab, c] of votes) if (c > bc) { bc = c; best = lab; }
    return best;
  });
}

// ============================ 自检（8 项） ============================
function selfTest() {
  const R = [];
  const rec = (name, ok, detail = '') => R.push({ name, ok: !!ok, detail });
  const rnd = rng(2024);

  // 1. Cholesky
  {
    const d = 6, A = zeros(d, d), B = zeros(d, d);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) B[i][j] = rnd.n();
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let t = 0; t < d; t++) s += B[i][t] * B[j][t]; A[i][j] = s + (i === j ? 5 : 0); }
    const L = cholesky(A);
    let e = 0;
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let t = 0; t < d; t++) s += L[i][t] * L[j][t]; e = Math.max(e, Math.abs(s - A[i][j])); }
    rec('Cholesky 重建 L·Lᵀ=A', e < 1e-8, `err=${e.toExponential(2)}`);
  }

  // 2. Jacobi 特征分解 + 正交
  {
    const d = 5, A = zeros(d, d), B = zeros(d, d);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) B[i][j] = rnd.n();
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let t = 0; t < d; t++) s += B[i][t] * B[j][t]; A[i][j] = s + (i === j ? 3 : 0); }
    const { values, vectors } = jacobiEig(A);
    let e = 0, o = 0;
    for (let j = 0; j < d; j++) {
      const v = vectors[j];
      for (let i = 0; i < d; i++) { let s = 0; for (let t = 0; t < d; t++) s += A[i][t] * v[t]; e = Math.max(e, Math.abs(s - values[j] * v[i])); }
      for (let k = 0; k < d; k++) { let s = 0; for (let t = 0; t < d; t++) s += vectors[j][t] * vectors[k][t]; o = Math.max(o, Math.abs(s - (j === k ? 1 : 0))); }
    }
    rec('Jacobi 特征残差 Aυ=λυ', e < 1e-8, `err=${e.toExponential(2)}`);
    rec('Jacobi 特征向量正交归一', o < 1e-8, `err=${o.toExponential(2)}`);
  }

  // 3. SVD 重建
  {
    const m = 6, n = 4, A = zeros(m, n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A[i][j] = rnd.n();
    const { U, s, Vt } = svd(A);
    const D = zeros(m, n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { let acc = 0; for (let t = 0; t < s.length; t++) acc += U[i][t] * s[t] * Vt[t][j]; D[i][j] = acc; }
    let e = 0; for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) e = Math.max(e, Math.abs(D[i][j] - A[i][j]));
    rec('SVD 重建 U·S·Vᵀ=A', e / (maxAbs(A) + 1e-12) < 1e-6, `rel=${(e / (maxAbs(A) + 1e-12)).toExponential(2)}`);
  }

  // 4. PCA：正交 + 与 SVD 秩-k 重建一致（Eckart–Young）
  {
    const n = 60, d = 4, X = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i][j] = rnd.n() * 2 + j;
    const k = 2;
    const model = pcaFit(X, k);
    let o = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { let s = 0; for (let t = 0; t < d; t++) s += model.components[a][t] * model.components[b][t]; o = Math.max(o, Math.abs(s - (a === b ? 1 : 0))); }
    const Xr = pcaInverse(model, pcaTransform(model, X));
    let ePca = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) ePca += Math.pow(Xr[i][j] - X[i][j], 2);
    // SVD 秩-k 截断重建（Eckart–Young 下界）
    const Xc = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Xc[i][j] = X[i][j] - model.mean[j];
    const S = svd(Xc);
    let eSvd = 0;
    for (let t = k; t < S.s.length; t++) eSvd += S.s[t] * S.s[t];
    const rel = Math.abs(ePca - eSvd) / (eSvd + 1e-12);
    rec('PCA 主成分正交归一', o < 1e-8, `err=${o.toExponential(2)}`);
    rec('PCA 重建误差 = Eckart–Young 下界', rel < 1e-6, `rel=${rel.toExponential(2)}`);
  }

  // 5. GMM 不变量
  {
    const n = 120, d = 3, K = 3, X = zeros(n, d);
    for (let i = 0; i < n; i++) { const c = i % K; for (let j = 0; j < d; j++) X[i][j] = rnd.n() * 0.8 + c * 6; }
    const g = gmmFit(X, K, { nIter: 40, seed: 7 });
    let rs = 0; for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < K; k++) s += g.resp[i][k]; rs = Math.max(rs, Math.abs(s - 1)); }
    let ws = 0; for (let k = 0; k < K; k++) ws += g.weights[k];
    let mono = true;
    for (let t = 1; t < g.loglik.length; t++) if (g.loglik[t] < g.loglik[t - 1] - 1e-6) mono = false;
    rec('GMM 责任度行和=1 & 权重和=1', rs < 1e-8 && Math.abs(ws - 1) < 1e-8, `resp=${rs.toExponential(1)}`);
    rec('GMM 对数似然单调 (EM 定理)', mono);
  }

  // 6. MLP 梯度检验
  {
    const n = 8, din = 3, h = 4, out = 3;
    const X = zeros(n, din); const y = new Array(n);
    for (let i = 0; i < n; i++) { for (let j = 0; j < din; j++) X[i][j] = rnd.n(); y[i] = i % out; }
    const W1 = zeros(din, h), b1 = zeros(1, h), W2 = zeros(h, out), b2 = zeros(1, out);
    for (let i = 0; i < din; i++) for (let j = 0; j < h; j++) W1[i][j] = rnd.n() * 0.3;
    for (let j = 0; j < h; j++) b1[0][j] = rnd.n() * 0.1;
    for (let i = 0; i < h; i++) for (let j = 0; j < out; j++) W2[i][j] = rnd.n() * 0.3;
    for (let j = 0; j < out; j++) b2[0][j] = rnd.n() * 0.1;
    const g = mlpBackward(W1, b1[0], W2, b2[0], X, y);
    const eps = 1e-5;
    const fd = (arr, i, j, grad) => {
      const o = arr[i][j];
      arr[i][j] = o + eps; const lp = mlpLoss(W1, b1[0], W2, b2[0], X, y);
      arr[i][j] = o - eps; const lm = mlpLoss(W1, b1[0], W2, b2[0], X, y);
      arr[i][j] = o;
      const num = (lp - lm) / (2 * eps);
      return Math.abs(num - grad[i][j]) / Math.max(1e-8, Math.abs(num) + Math.abs(grad[i][j]));
    };
    let mx = 0;
    for (let i = 0; i < din; i++) for (let j = 0; j < h; j++) mx = Math.max(mx, fd(W1, i, j, g.gW1));
    for (let j = 0; j < h; j++) mx = Math.max(mx, fd(b1, 0, j, [g.gb1]));
    for (let i = 0; i < h; i++) for (let j = 0; j < out; j++) mx = Math.max(mx, fd(W2, i, j, g.gW2));
    for (let j = 0; j < out; j++) mx = Math.max(mx, fd(b2, 0, j, [g.gb2]));
    rec('MLP 反向传播梯度检验', mx < 1e-6, `maxRelErr=${mx.toExponential(2)}`);
  }

  // 7. Attention 梯度检验
  {
    const n = 5, d = 4, r2 = rng(11);
    const Q = zeros(n, d), K = zeros(n, d), V = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { Q[i][j] = r2.n(); K[i][j] = r2.n(); V[i][j] = r2.n(); }
    const cache = attention(Q, K, V);
    const T = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) T[i][j] = r2.n();
    let loss = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) loss += 0.5 * Math.pow(cache.out[i][j] - T[i][j], 2);
    const dOut = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) dOut[i][j] = cache.out[i][j] - T[i][j];
    const { dQ, dK, dV } = attentionBackward(dOut, cache);
    const eps = 1e-5;
    const lossOf = () => { let s = 0; const o = attention(Q, K, V).out; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) s += 0.5 * Math.pow(o[i][j] - T[i][j], 2); return s; };
    const fd = (M, G) => {
      let worst = 0;
      for (let i = 0; i < M.length; i++) for (let j = 0; j < d; j++) {
        const o = M[i][j];
        M[i][j] = o + eps; const lp = lossOf();
        M[i][j] = o - eps; const lm = lossOf();
        M[i][j] = o;
        const num = (lp - lm) / (2 * eps);
        worst = Math.max(worst, Math.abs(num - G[i][j]) / Math.max(1e-8, Math.abs(num) + Math.abs(G[i][j])));
      }
      return worst;
    };
    const mx = Math.max(fd(Q, dQ), fd(K, dK), fd(V, dV));
    rec('Attention 反向传播梯度检验', mx < 1e-6, `maxRelErr=${mx.toExponential(2)}`);
  }

  // 8. kNN：k=1 训练集零误差
  {
    const n = 40, d = 3, X = zeros(n, d); const y = new Array(n);
    for (let i = 0; i < n; i++) { for (let j = 0; j < d; j++) X[i][j] = rnd.n(); y[i] = i % 2; }
    const pred = knnPredict(X, y, X, 1);
    let acc = 0; for (let i = 0; i < n; i++) if (pred[i] === y[i]) acc++;
    rec('kNN k=1 训练集零误差（过拟合边界）', acc === n, `acc=${acc}/${n}`);
  }

  return R;
}

const ForgeAI = {
  zeros, eye, clone, transpose, matmul, dot, cholesky, jacobiEig, svd,
  pcaFit, pcaTransform, pcaInverse, gmmFit,
  mlpForward, mlpBackward, mlpLoss, attention, attentionBackward,
  knnPredict, selfTest,
};

if (typeof module !== 'undefined' && module.exports) module.exports = ForgeAI;

if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
  const R = selfTest();
  const pass = R.filter(x => x.ok).length;
  const line = '='.repeat(74);
  console.log(line);
  console.log(' forge.js 自检报告  | forge-ai JS 内核  | 作者 晨星');
  console.log(line);
  for (const t of R) {
    console.log(` ${t.ok ? '✅' : '❌'}  ${t.name}${t.detail ? `   [${t.detail}]` : ''}`);
  }
  console.log('-'.repeat(74));
  console.log(` 通过 ${pass}/${R.length}`);
  console.log(line);
  process.exit(pass === R.length ? 0 : 1);
}
