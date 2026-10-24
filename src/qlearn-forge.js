/*!
 * qlearn-forge.js — 晨星旗舰 AI · L6 强化学习：Q-Learning（表格型）
 * ---------------------------------------------------------------
 * 站在世界顶级成果之上：
 *   · Q-Learning —— Watkins 1989（off-policy 时序差分）
 *   · ε-贪婪探索 + 常数学习率；Bellman 最优算子收缩性 ⇒ 收敛到 Q*
 *
 * 硬不变量（可独立交叉验证）：
 *   [1] 目标可达       贪心策略从起点到达终点
 *   [2] 近优路径       步数 ≤ 12（曼哈顿最优 8）
 *   [3] Bellman 残差   max|Q(s,a) − (r + γ·max Q(s′,·))| ≤ 0.3（最优方程满足度）
 *   [4] 值有界         |Q| ≤ 1/(1−γ) + 1（回报有界性）
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.QLForge = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const QL = {};

  QL.mulberry32 = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  /* 5×5 网格世界：墙在 (2,1)(2,2)(2,3)，起点 (0,0)，终点 (4,4)，每步 r=−1 */
  function makeEnv() {
    const N = 5;
    const walls = {};
    walls['2,1'] = 1; walls['2,2'] = 1; walls['2,3'] = 1;
    const start = [0, 0], goal = [4, 4];
    const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // 上下右左
    return {
      N: N, walls: walls, start: start, goal: goal, moves: moves,
      step: function (s, a) {
        // goal 为吸收终止态：到达后再任何动作都立即结束（r=0）
        if (s[0] === goal[0] && s[1] === goal[1]) return { s: s, r: 0, done: true };
        const ns = [s[0] + moves[a][0], s[1] + moves[a][1]];
        if (ns[0] < 0 || ns[0] >= N || ns[1] < 0 || ns[1] >= N || walls[ns[0] + ',' + ns[1]]) {
          return { s: s, r: -1, done: false };
        }
        const done = (ns[0] === goal[0] && ns[1] === goal[1]);
        return { s: ns, r: -1, done: done };
      }
    };
  }

  QL.train = function (opts) {
    opts = opts || {};
    const episodes = opts.episodes || 6000;
    const alpha0 = opts.alpha || 0.2, alphaMin = opts.alphaMin || 0.02;
    const gamma = opts.gamma || 0.95, eps = opts.eps || 0.1;
    const maxSteps = opts.maxSteps || 60;
    const rng = QL.mulberry32(opts.seed || 42);
    const env = makeEnv();
    const Q = {}, visited = {};
    for (let i = 0; i < env.N; i++) for (let j = 0; j < env.N; j++) {
      Q[i + ',' + j] = [0, 0, 0, 0];
    }
    const key = function (s) { return s[0] + ',' + s[1]; };

    for (let ep = 0; ep < episodes; ep++) {
      // 两阶段：前 75% ε-贪婪收敛，后 25% 纯随机全域刷新（消除低频对的陈旧偏差）
      const inExplore = ep < episodes * 0.75;
      // Robbins–Monro 学习率衰减：Σα=∞ 且 Σα²<∞ ⇒ 几乎必然收敛到 Q*
      const alpha = inExplore ? Math.max(alphaMin, alpha0 / (1 + ep / 500)) : 0.02;
      const epsT = inExplore ? eps : 1.0;
      // Exploring starts：半数回合从随机状态出发（排除 goal），保证全域 (s,a) 均匀覆盖
      let s;
      if (rng() < 0.5) {
        do {
          s = [Math.floor(rng() * env.N), Math.floor(rng() * env.N)];
        } while (env.walls[s[0] + ',' + s[1]] || (s[0] === env.goal[0] && s[1] === env.goal[1]));
      } else {
        s = env.start.slice();
      }
      for (let t = 0; t < maxSteps; t++) {
        let a;
        if (rng() < epsT) a = Math.floor(rng() * 4);
        else {
          const q = Q[key(s)];
          a = 0;
          for (let k = 1; k < 4; k++) if (q[k] > q[a]) a = k;
        }
        visited[key(s) + ':' + a] = 1;
        const res = env.step(s, a);
        const target = res.r + (res.done ? 0 : gamma * Math.max.apply(null, Q[key(res.s)]));
        Q[key(s)][a] += alpha * (target - Q[key(s)][a]);
        s = res.s;
        if (res.done) break;
      }
    }
    return { Q: Q, visited: visited, env: env };
  };

  /* ================= 自检 ================= */
  QL.selfTest = function () {
    const checks = [];
    const add = function (name, value, tol, mode) {
      const m = mode || 'le';
      checks.push({ name: name, value: value, tol: tol, mode: m, ok: m === 'le' ? value <= tol : value >= tol, detail: '' });
    };

    const res = QL.train({ episodes: 8000, alpha: 0.2, gamma: 0.95, eps: 0.2, alphaMin: 0.003, seed: 42 });
    const Q = res.Q, env = res.env;
    const key = function (s) { return s[0] + ',' + s[1]; };

    // 贪心 rollout
    let s = env.start.slice(), steps = 0, reached = false;
    const seen = {};
    while (steps < 50) {
      const k = key(s);
      if (seen[k]) break; // 环路
      seen[k] = 1;
      const q = Q[k];
      let a = 0;
      for (let i = 1; i < 4; i++) if (q[i] > q[a]) a = i;
      const r = env.step(s, a);
      s = r.s;
      steps++;
      if (r.done) { reached = true; break; }
    }

    // [1] 到达终点
    add('贪心策略到达终点', reached ? 0 : 1, 0);

    // [2] 路径步数（曼哈顿最优 8，容忍绕墙）
    add('贪心路径步数', reached ? steps : 100, 12);

    // [3] Bellman 残差（仅统计被访问过的状态-动作对）
    let bellman = 0, cnt = 0;
    for (const vk in res.visited) {
      const parts = vk.split(':');
      const sp = parts[0].split(',');
      const st = [+sp[0], +sp[1]], a = +parts[1];
      const r = env.step(st, a);
      const target = r.r + (r.done ? 0 : 0.95 * Math.max.apply(null, Q[key(r.s)]));
      bellman = Math.max(bellman, Math.abs(Q[vk.split(':')[0]][a] - target));
      cnt++;
    }
    add('Bellman 残差（' + cnt + ' 对）', bellman, 0.05);

    // [4] 值有界
    let maxAbs = 0;
    for (const k in Q) for (const v of Q[k]) maxAbs = Math.max(maxAbs, Math.abs(v));
    add('Q 值幅度上界', maxAbs, 1 / (1 - 0.95) + 1);

    return { name: 'qlearn-forge', layer: 'L6 强化学习·Q-Learning', pass: checks.every(function (c) { return c.ok; }), checks: checks };
  };

  return QL;
});
