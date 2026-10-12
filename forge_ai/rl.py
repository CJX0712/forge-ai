"""forge_ai.rl —— 从零实现的强化学习（表格 Q-learning + 网格世界 oracle）。

集大成：最优值函数以从零价值迭代（Bellman 方程不动点）作为 oracle。
零重写：与价值迭代的*最优策略*交叉验证。

数学不变量：
- QLearning :  Bellman 残差 max|Q − (r + γ·max Q')| → 0（收敛到最优）
-          :  贪心策略与价值迭代最优策略一致
"""
from __future__ import annotations
import numpy as np

__all__ = ["GridWorld", "QLearning"]


class GridWorld:
    def __init__(self, n=4, goal=(3, 3), step_cost=-1.0, gamma=0.99):
        self.n = n
        self.goal = goal
        self.step_cost = step_cost
        self.gamma = gamma
        self.n_states = n * n
        self.n_actions = 4                      # 0上 1下 2左 3右
        self.delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _coord(self, s):
        return divmod(s, self.n)

    def step(self, s, a):
        r, c = self._coord(s)
        dr, dc = self.delta[a]
        nr, nc = min(max(r + dr, 0), self.n - 1), min(max(c + dc, 0), self.n - 1)
        s2 = nr * self.n + nc
        done = (nr, nc) == self.goal
        reward = 0.0 if done else self.step_cost
        return s2, reward, done

    def value_iteration(self, tol=1e-9, max_iter=1000):
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                if self._coord(s) == self.goal:
                    V_new[s] = 0.0
                    continue
                best = -np.inf
                for a in range(self.n_actions):
                    s2, r, done = self.step(s, a)
                    v = r if done else r + self.gamma * V[s2]
                    best = max(best, v)
                V_new[s] = best
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            if self._coord(s) == self.goal:
                continue
            best, ba = -np.inf, 0
            for a in range(self.n_actions):
                s2, r, done = self.step(s, a)
                v = r if done else r + self.gamma * V[s2]
                if v > best:
                    best, ba = v, a
            policy[s] = ba
        return V, policy


class QLearning:
    def __init__(self, world, lr=0.1, eps=0.15, seed=0):
        self.world = world
        self.lr = lr
        self.eps = eps
        self.seed = seed
        self.Q = np.zeros((world.n_states, world.n_actions))
        self.bellman_residuals_ = []

    def train(self, n_episodes=8000):
        rng = np.random.default_rng(self.seed)
        for _ in range(n_episodes):
            s = rng.integers(self.world.n_states)
            while self.world._coord(s) != self.world.goal:
                a = (rng.integers(self.world.n_actions) if rng.random() < self.eps
                     else int(np.argmax(self.Q[s])))
                s2, r, done = self.world.step(s, a)
                target = r if done else r + self.world.gamma * self.Q[s2].max()
                self.Q[s, a] += self.lr * (target - self.Q[s, a])
                if done:
                    break
                s = s2
        # Bellman 方程只定义于非终止态：终点态 Q 恒为 0 且训练不更新，须屏蔽
        res = 0.0
        for s in range(self.world.n_states):
            if self.world._coord(s) == self.world.goal:
                continue
            for a in range(self.world.n_actions):
                s2, r, done = self.world.step(s, a)
                target = r if done else r + self.world.gamma * self.Q[s2].max()
                res = max(res, abs(self.Q[s, a] - target))
        self.bellman_residuals_.append(float(res))
        return self

    @property
    def bellman_residual(self):
        return self.bellman_residuals_[-1] if self.bellman_residuals_ else None

    def greedy_policy(self):
        policy = np.zeros(self.world.n_states, dtype=int)
        for s in range(self.world.n_states):
            if self.world._coord(s) == self.world.goal:
                continue
            policy[s] = int(np.argmax(self.Q[s]))
        return policy

    def policy_matches(self, optimal_policy):
        lp = self.greedy_policy()
        mask = np.array([self.world._coord(s) != self.world.goal
                         for s in range(self.world.n_states)])
        return float(np.mean(lp[mask] == optimal_policy[mask]))
