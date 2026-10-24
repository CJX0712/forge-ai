/*!
 * math-forge.js — 晨星旗舰 AI · L0 数学内核
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上的零依赖数值内核：
 *   · 对称特征分解   —— Jacobi 循环旋转（Jacobi 1846；Golub & Van Loan《Matrix Computations》）
 *   · SPD 直接求解   —— Cholesky 分解（O(n³/3)，半带宽内最优直接法）
 *   · 可复现随机     —— mulberry32 + Box–Muller（同种子 ⇒ 同随机流）
 *   · 梯度检验       —— 中心差分金标准（maxRelErr ≈ 1e-7）
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 正交归一      ‖VᵀV − I‖ ≤ 1e-14
 *   [2] 特征残差      max‖A·vᵢ − λᵢ·vᵢ‖ ≤ 1e-12
 *   [3] 谱定理        Σλᵢ = tr(A)
 *   [4] Cholesky 重构 ‖L·Lᵀ − A‖ ≤ 1e-12
 *   [5] 解残差        ‖A·x − b‖ ≤ 1e-10
 *   [6] 梯度检验      maxRelErr ≤ 1e-6（解析梯度 vs 中心差分）
 *   [7] 可复现性      同种子两次随机流完全一致
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MathForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const MF = {};

  /* ================= 可复现随机 ================= */
  MF.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  MF.randn = function (rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  /* ================= 矩阵基元 ================= */
  MF.zeros = function (r, c) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
    return m;
  };
  MF.identity = function (n) {
    const m = MF.zeros(n, n);
    for (let i = 0; i < n; i++) m[i][i] = 1;
    return m;
  };
  MF.transpose = function (A) {
    const r = A.length, c = A[0].length, T = MF.zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  };
  MF.matMul = function (A, B) {
    const r = A.length, k = B.length, c = B[0].length, C = MF.zeros(r, c);
    for (let i = 0; i < r; i++)
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        for (let j = 0; j < c; j++) C[i][j] += a * B[p][j];
      }
    return C;
  };
  MF.matVec = function (A, x) {
    const r = A.length, c = x.length, y = new Array(r).fill(0);
    for (let i = 0; i < r; i++) {
      let s = 0;
      for (let j = 0; j < c; j++) s += A[i][j] * x[j];
      y[i] = s;
    }
    return y;
  };
  MF.frobeniusNorm = function (A) {
    let s = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) s += A[i][j] * A[i][j];
    return Math.sqrt(s);
  };
  MF.maxAbs = function (A) {
    let m = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j]));
    return m;
  };
  MF.vecNorm = function (x) { let s = 0; for (const v of x) s += v * v; return Math.sqrt(s); };

  /* ================= Cholesky（SPD 直接求解器） ================= */
  MF.cholesky = function (A) {
    const n = A.length, L = MF.zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
      }
    return L;
  };
  MF.choleskySolve = function (L, b) {
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
  };
  MF.solveSPD = function (A, b) { return MF.choleskySolve(MF.cholesky(A), b); };

  /* ================= Jacobi 对称特征分解（循环旋转） ================= */
  MF.jacobiEigenSym = function (A, opts) {
    opts = opts || {};
    const tol = opts.tol != null ? opts.tol : 1e-13;
    const maxSweeps = opts.maxSweeps || 100;
    const n = A.length;
    const a = A.map(function (r) { return r.slice(); });
    const V = MF.identity(n);
    const maxOffDiag = function () {
      let m = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m = Math.max(m, Math.abs(a[i][j]));
      return m;
    };
    for (let sweep = 0; sweep < maxSweeps && maxOffDiag() > tol; sweep++) {
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
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
            a[p][k] = a[k][p];
            a[q][k] = a[k][q];
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
    }
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(i);
    idx.sort(function (i, j) { return a[j][j] - a[i][i]; });
    return {
      values: idx.map(function (i) { return a[i][i]; }),
      vectors: MF.identity(n).map(function (_, r) { return idx.map(function (i) { return V[r][i]; }); })
    };
  };

  /* ================= 梯度检验（中心差分金标准） ================= */
  MF.gradientCheck = function (f, g, x, eps) {
    eps = eps || 1e-5;
    let maxRel = 0;
    for (let i = 0; i < x.length; i++) {
      const xp = x.slice(), xm = x.slice();
      xp[i] += eps; xm[i] -= eps;
      const num = (f(xp) - f(xm)) / (2 * eps);
      const ana = g(x)[i];
      const rel = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
      if (rel > maxRel) maxRel = rel;
    }
    return maxRel;
  }

  /* ================= 自检 ================= */
  MF.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode, detail) {
      const m = mode || 'le';
      checks.push({
        name: name, value: value, tol: tol, mode: m,
        ok: m === 'le' ? value <= tol : value >= tol,
        detail: detail || ''
      });
    };

    // [1] 矩阵乘法 vs 暴力三重循环
    const rng = MF.mulberry32(20261024);
    const A = [], B = [];
    for (let i = 0; i < 3; i++) { A.push([]); for (let j = 0; j < 4; j++) A[i].push(MF.randn(rng)); }
    for (let i = 0; i < 4; i++) { B.push([]); for (let j = 0; j < 2; j++) B[i].push(MF.randn(rng)); }
    const C = MF.matMul(A, B);
    let mmErr = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[i][k] * B[k][j];
      mmErr = Math.max(mmErr, Math.abs(s - C[i][j]));
    }
    add('矩阵乘法 vs 暴力枚举', mmErr, 1e-12);

    // [2] 转置不变性
    const ATT = MF.transpose(MF.transpose(A));
    let trErr = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) trErr = Math.max(trErr, Math.abs(ATT[i][j] - A[i][j]));
    add('转置不变性 ‖(Aᵀ)ᵀ−A‖', trErr, 0);

    // [3] Jacobi 解析解交叉验证 [[2,1],[1,2]] → λ={3,1}
    const e23 = MF.jacobiEigenSym([[2, 1], [1, 2]]);
    const lamErr = Math.max(Math.abs(e23.values[0] - 3), Math.abs(e23.values[1] - 1));
    add('Jacobi 解析解 λ={3,1}', lamErr, 1e-12);

    // [4] 随机对称阵：正交归一 + 特征残差 + 谱定理
    const n = 6, S = MF.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = MF.randn(rng); S[i][j] = v; S[j][i] = v; }
    const eig = MF.jacobiEigenSym(S);
    const V = eig.vectors;
    const VtV = MF.matMul(MF.transpose(V), V);
    const I = MF.identity(n);
    let orthErr = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) orthErr = Math.max(orthErr, Math.abs(VtV[i][j] - I[i][j]));
    add('正交归一 ‖VᵀV−I‖', orthErr, 1e-14);
    let resErr = 0;
    for (let k = 0; k < n; k++) {
      const vk = V.map(function (row) { return row[k]; });
      const Av = MF.matVec(S, vk);
      for (let i = 0; i < n; i++) resErr = Math.max(resErr, Math.abs(Av[i] - eig.values[k] * vk[i]));
    }
    add('特征残差 ‖Av−λv‖', resErr, 1e-12);
    let tr = 0, lamSum = 0;
    for (let i = 0; i < n; i++) { tr += S[i][i]; lamSum += eig.values[i]; }
    add('谱定理 Σλ=tr(A)', Math.abs(lamSum - tr), 1e-11);

    // [5] Cholesky 重构 + 解残差
    const SPD = MF.matMul(MF.transpose(S), S);
    for (let i = 0; i < n; i++) SPD[i][i] += 0.5;
    const L = MF.cholesky(SPD);
    const LLt = MF.matMul(L, MF.transpose(L));
    let cholErr = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cholErr = Math.max(cholErr, Math.abs(LLt[i][j] - SPD[i][j]));
    add('Cholesky 重构 ‖LLᵀ−A‖', cholErr, 1e-12);
    const b = []; for (let i = 0; i < n; i++) b.push(MF.randn(rng));
    const x = MF.solveSPD(SPD, b);
    const Ax = MF.matVec(SPD, x);
    let solErr = 0;
    for (let i = 0; i < n; i++) solErr = Math.max(solErr, Math.abs(Ax[i] - b[i]));
    add('Cholesky 解残差 ‖Ax−b‖', solErr, 1e-10);

    // [6] 梯度检验：f(x)=Σxᵢ² + x₀x₁x₂
    const f = function (x) { let s = 0; for (const v of x) s += v * v; return s + x[0] * x[1] * x[2]; };
    const gf = function (x) { return [2 * x[0] + x[1] * x[2], 2 * x[1] + x[0] * x[2], 2 * x[2] + x[0] * x[1]]; };
    add('梯度检验 maxRelErr', MF.gradientCheck(f, gf, [0.7, -1.3, 2.1], 1e-5), 1e-6);

    // [7] 可复现性
    const r1 = MF.mulberry32(42), r2 = MF.mulberry32(42);
    let repErr = 0;
    for (let i = 0; i < 100; i++) repErr = Math.max(repErr, Math.abs(MF.randn(r1) - MF.randn(r2)));
    add('随机流可复现（同种子）', repErr, 0);

    return { name: 'math-forge', layer: 'L0 数学内核', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return MF;
});
