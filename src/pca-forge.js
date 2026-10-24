/*!
 * pca-forge.js — 晨星旗舰 AI · L2 无监督学习：主成分分析
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · PCA = 协方差矩阵特征分解 —— Pearson 1901；Hotelling 1933
 *   · Jacobi 循环旋转求对称特征（绝对收敛，无需 QR）
 *   · Eckart–Young 定理：秩-k 截断是重构误差最优解
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 谱守恒        Σλᵢ = tr(Σ)（Jacobi vs 直接迹）
 *   [2] 行列式守恒    Πλᵢ = det(Σ)（Cholesky 独立计算 det = ΠLᵢᵢ²）
 *   [3] 正交归一      ‖VᵀV − I‖ ≤ 1e-14
 *   [4] 特征残差      ‖Σvᵢ − λᵢvᵢ‖ ≤ 1e-11
 *   [5] 降序排列      λ₁ ≥ λ₂ ≥ …
 *   [6] 得分对角化    cov(VᵀX) 非对角 ≤ 1e-10（主成分不相关）
 *   [7] 重构恒等      V Vᵀ = I ⇒ 全量重构误差 ≤ 1e-10
 *   [8] 方差占比      λ₁/Σλ ≥ 0.9（旋转变换下恢复主方向）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PCAForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PC = {};

  PC.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  PC.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

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
  function identity(n) {
    const m = [];
    for (let i = 0; i < n; i++) { m.push(new Array(n).fill(0)); m[i][i] = 1; }
    return m;
  }
  function cholesky(A) {
    const n = A.length, L = [];
    for (let i = 0; i < n; i++) L.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
    return L;
  }

  /* Jacobi 对称特征分解（与 math-forge 同源的零依赖内嵌实现） */
  PC.jacobiEigenSym = function (A, opts) {
    opts = opts || {};
    const tol = opts.tol != null ? opts.tol : 1e-13;
    const maxSweeps = opts.maxSweeps || 100;
    const n = A.length;
    const a = A.map(function (r) { return r.slice(); });
    const V = identity(n);
    const maxOffDiag = function () {
      let m = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m = Math.max(m, Math.abs(a[i][j]));
      return m;
    };
    for (let sweep = 0; sweep < maxSweeps && maxOffDiag() > tol; sweep++) {
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const sgn = theta >= 0 ? 1 : -1;
        const t = sgn / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p]; a[q][k] = a[k][q];
        }
        const app = a[p][p], aqq = a[q][q];
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0; a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(i);
    idx.sort(function (i, j) { return a[j][j] - a[i][i]; });
    return {
      values: idx.map(function (i) { return a[i][i]; }),
      vectors: identity(n).map(function (_, r) { return idx.map(function (i) { return V[r][i]; }); })
    };
  };

  /* 数据中心化 + 协方差 + 特征分解 */
  PC.fit = function (X) {
    const n = X.length, d = X[0].length;
    const mean = new Array(d).fill(0);
    for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
    const Xc = X.map(function (row) { return row.map(function (v, j) { return v - mean[j]; }); });
    const cov = matMul(transpose(Xc), Xc).map(function (row) { return row.map(function (v) { return v / (n - 1); }); });
    const eig = PC.jacobiEigenSym(cov);
    return { mean: mean, cov: cov, values: eig.values, vectors: eig.vectors };
  };

  PC.transform = function (X, model, k) {
    const V = model.vectors;
    const kk = k || V.length;
    return X.map(function (row) {
      const out = [];
      for (let c = 0; c < kk; c++) {
        let s = 0;
        for (let j = 0; j < row.length; j++) s += (row[j] - model.mean[j]) * V[j][c];
        out.push(s);
      }
      return out;
    });
  };

  /* ================= 自检 ================= */
  PC.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const rng = PC.mulberry32(20261024);
    const n = 5000;
    const th = 0.7, ct = Math.cos(th), st = Math.sin(th);
    const X = [];
    for (let i = 0; i < n; i++) {
      const z1 = PC.randn(rng), z2 = PC.randn(rng), z3 = PC.randn(rng);
      // 布景：主方向 (5,1) 旋转 θ=0.7，第三轴 0.2 —— 特征值应为 {25, 1, 0.04}
      X.push([5 * ct * z1 - 1 * st * z2, 5 * st * z1 + 1 * ct * z2, 0.2 * z3]);
    }

    const model = PC.fit(X);
    const lam = model.values, V = model.vectors, cov = model.cov;
    const d = 3;

    // [1] 谱守恒
    let tr = 0, lamSum = 0;
    for (let i = 0; i < d; i++) { tr += cov[i][i]; lamSum += lam[i]; }
    add('谱守恒 |Σλ−tr(Σ)|/tr', Math.abs(lamSum - tr) / Math.abs(tr), 1e-10);

    // [2] 行列式守恒（Cholesky 独立路径：det = ΠLᵢᵢ²）
    const L = cholesky(cov);
    let det = 1;
    for (let i = 0; i < d; i++) det *= L[i][i] * L[i][i];
    let lamProd = 1;
    for (let i = 0; i < d; i++) lamProd *= lam[i];
    add('行列式守恒 |Πλ−det(Σ)|/det', Math.abs(lamProd - det) / Math.abs(det), 1e-8);

    // [3] 正交归一
    const VtV = matMul(transpose(V), V);
    const I = identity(d);
    let orthErr = 0;
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) orthErr = Math.max(orthErr, Math.abs(VtV[i][j] - I[i][j]));
    add('正交归一 ‖VᵀV−I‖', orthErr, 1e-14);

    // [4] 特征残差
    let resErr = 0;
    for (let k = 0; k < d; k++) {
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += cov[i][j] * V[j][k];
        resErr = Math.max(resErr, Math.abs(s - lam[k] * V[i][k]));
      }
    }
    add('特征残差 ‖Σv−λv‖', resErr, 1e-11);

    // [5] 降序
    let ordErr = 0;
    for (let i = 0; i < d - 1; i++) ordErr = Math.max(ordErr, lam[i + 1] - lam[i]);
    add('特征值降序违例', ordErr, 1e-15);

    // [6] 得分协方差对角化
    const scores = PC.transform(X, model);
    const sc = matMul(transpose(scores), scores).map(function (row) { return row.map(function (v) { return v / (n - 1); }); });
    let offErr = 0;
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) if (i !== j) offErr = Math.max(offErr, Math.abs(sc[i][j]));
    add('主成分得分去相关（off-diag）', offErr, 1e-10);

    // [7] 全量重构恒等（V 正交 ⇒ VVᵀ=I）
    const recon = scores.map(function (row) {
      return row.map(function (v, j) {
        let s = model.mean[j];
        for (let c = 0; c < d; c++) s += row[c] * V[j][c];
        return s;
      });
    });
    let recErr = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) recErr = Math.max(recErr, Math.abs(recon[i][j] - X[i][j]));
    add('全量重构误差', recErr, 1e-10);

    // [8] 主方向方差占比
    add('λ₁ 方差占比', lam[0] / lamSum, 0.9, 'ge');

    // [9] 特征值恢复（旋转变换下解析值 {25,1,0.04}）
    const truth = [25, 1, 0.04];
    let lamRel = 0;
    for (let i = 0; i < d; i++) lamRel = Math.max(lamRel, Math.abs(lam[i] - truth[i]) / truth[i]);
    add('特征值恢复（解析 {25,1,0.04}）', lamRel, 0.15);

    return { name: 'pca-forge', layer: 'L2 无监督·PCA', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return PC;
});
