#!/usr/bin/env node
/*!
 * test-runner.js — 晨星旗舰 AI · 无头 Node 自检总控
 * ---------------------------------------------------------------
 * 用法：  node src/test-runner.js
 * 退出码：0 = 全绿；1 = 存在失败
 *
 * 每个 forge 模块自带 selfTest()，返回可独立交叉验证的「硬不变量」清单。
 * 本文件只做编排与汇总，不参与任何判定逻辑。
 *
 * 作者：晨星（CJX0712）  License：MIT
 */
'use strict';

const path = require('path');

const MODULES = [
  ['L0 数学内核', './math-forge.js'],
  ['L1 监督·线性回归', './linreg-forge.js'],
  ['L1 监督·逻辑回归', './logreg-forge.js'],
  ['L1 监督·决策树 CART', './tree-forge.js'],
  ['L2 无监督·PCA', './pca-forge.js'],
  ['L2 无监督·K-Means', './kmeans-forge.js'],
  ['L2 无监督·GMM/EM', './gmm-forge.js'],
  ['L4 深度学习·MLP', './nn-forge.js'],
  ['L5 注意力·Transformer', './attn-forge.js'],
  ['L6 强化学习·Q-Learning', './qlearn-forge.js']
];

function fmt(v) {
  if (v === 0) return '0';
  if (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e12) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(6);
}

function pad(str, width) {
  let w = 0;
  for (const ch of str) w += ch.charCodeAt(0) > 0x2E80 ? 2 : 1;
  return str + ' '.repeat(Math.max(1, width - w));
}

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║   晨星旗舰 AI · Forge AI —— 无头 Node 全量自检                  ║');
console.log('║   零依赖 · 可复现 · 硬不变量可独立交叉验证   作者：晨星          ║');
console.log('╚════════════════════════════════════════════════════════════════╝');

let passMods = 0, passChecks = 0, totalChecks = 0;
const failures = [];

for (const entry of MODULES) {
  const name = entry[0], rel = entry[1];
  let r;
  try {
    const mod = require(path.join(__dirname, rel));
    r = mod.selfTest();
  } catch (e) {
    console.log('');
    console.log('❌ ' + name + '  模块异常: ' + e.message);
    failures.push(name + ' → ' + e.message);
    continue;
  }
  const okc = r.checks.filter(function (c) { return c.ok; }).length;
  totalChecks += r.checks.length;
  passChecks += okc;
  if (r.pass) passMods++;

  console.log('');
  console.log((r.pass ? '✅ ' : '❌ ') + name + '  ' + okc + '/' + r.checks.length);
  for (const c of r.checks) {
    const op = c.mode === 'ge' ? ' ≥ ' : ' ≤ ';
    console.log('   ' + (c.ok ? '✅' : '❌') + ' ' + pad(c.name, 34) + pad(fmt(c.value), 14) + op + fmt(c.tol));
    if (!c.ok) failures.push(name + ' / ' + c.name + ' （实测 ' + fmt(c.value) + op + fmt(c.tol) + '）');
  }
}

console.log('');
console.log('─────────────────────────────────────────────────────────────────');
const allGreen = (passMods === MODULES.length && passChecks === totalChecks);
console.log('模块 ' + passMods + '/' + MODULES.length + '    检查项 ' + passChecks + '/' + totalChecks +
  (allGreen ? '    🎉 全绿（FLASHIP SELF-TEST FULLY GREEN）' : '    ❌ 存在失败'));
console.log('─────────────────────────────────────────────────────────────────');
if (failures.length) {
  console.log('失败明细：');
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
process.exit(0);
