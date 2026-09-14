#!/usr/bin/env node
/**
 * 诊断：验证信息层级是否在"因果掌握态"口径下成立
 *  obs(可观察代理≤因果掌握态) ≤ pre(作答前BKT掌握度) ≤ oracle(pOk)
 * 澄清 [R2] 失败根因：mastery_tgt[k] 是答完第 k 题后的后验（用到了 corrects[k] 标签）→ 泄漏。
 * 因果掌握态进入第 k 题 = mastery_tgt[k-1]（k≥1），k=0 用 prior=0.35。
 */
const fs = require('fs');
const path = require('path');
const DATA = path.join(__dirname, '..', '..', 'training', 'data', 'p0_prereq.jsonl');

function rankAUC(p, y) {
  const pairs = [];
  for (let i = 0; i < p.length; i++) if (y[i] === 0 || y[i] === 1) pairs.push([p[i], y[i]]);
  const m = pairs.length;
  if (m === 0) return null;
  let npos = 0; for (const [, yy] of pairs) npos += yy;
  const nneg = m - npos;
  if (npos === 0 || nneg === 0) return null;
  pairs.sort((a, b) => a[0] - b[0]);
  let sumPosRanks = 0, i = 0;
  while (i < m) {
    let j = i;
    while (j < m && pairs[j][0] === pairs[i][0]) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (pairs[k][1] === 1) sumPosRanks += avgRank;
    i = j;
  }
  return (sumPosRanks - npos * (npos + 1) / 2) / (npos * nneg);
}

const rows = fs.readFileSync(DATA, 'utf8').split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l));

const PRIOR = 0.35;
let hasP = 0, nNull = 0;
const obs = { p: [], y: [] }, pre = { p: [], y: [] }, orc = { p: [], y: [] }, leak = { p: [], y: [] }, preNext = { p: [], y: [] };

for (const r of rows) {
  const c = r.corrects, mt = r.mastery_tgt, po = r.pOk;
  if (!po) { nNull++; continue; }
  for (let t = 0; t < c.length; t++) {
    if (po[t] == null) continue;
    hasP++;
    const y = c[t];
    // obs：t 之前的答对率（因果）
    const hist = t === 0 ? [] : c.slice(0, t);
    obs.p.push(hist.length ? hist.reduce((a, b) => a + b, 0) / hist.length : 0.5); obs.y.push(y);
    // pre：进入第 t 题的掌握度（因果）—— k=0 用 prior，否则用 答完 t-1 题的后验 或 进入态
    pre.p.push(t === 0 ? PRIOR : (Number.isFinite(mt[t - 1]) ? mt[t - 1] : PRIOR)); pre.y.push(y);
    // leak：答完第 t 题的后验（泄标签，应 > oracle 暴露泄漏）
    leak.p.push(Number.isFinite(mt[t]) ? mt[t] : 0.5); leak.y.push(y);
    // oracle
    orc.p.push(po[t]); orc.y.push(y);
  }
}

const A = (x) => rankAUC(x.p, x.y).toFixed(4);
console.log('nNull 行(无pOk)=', nNull, ' 参与评测位置=', hasP);
console.log('obs  (因果答对率)   =', A(obs));
console.log('pre  (因果BKT进入态)=', A(pre));
console.log('leak (答后验·泄标签) =', A(leak));
console.log('orc  (pOk true概率) =', A(orc));
console.log('---');
console.log('层级 (obs ≤ pre ≤ orc)?',
  rankAUC(obs.p, obs.y) <= rankAUC(pre.p, pre.y) + 1e-9,
  '| orc ≥ pre?', rankAUC(orc.p, orc.y) >= rankAUC(pre.p, pre.y) - 1e-9,
  '| orc ≥ obs?', rankAUC(orc.p, orc.y) >= rankAUC(obs.p, obs.y) - 1e-9);
console.log('泄漏证据：leak(', A(leak), ') > orc(', A(orc), ')?', rankAUC(leak.p, leak.y) > rankAUC(orc.p, orc.y));