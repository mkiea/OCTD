#!/usr/bin/env node
/**
 * 模拟器诊断脚本（一次性，改完即弃；不入 test/ 资产）
 * =====================================================
 * 为论文"① 模拟器是否有 bug"输出可核对的数值证据，并顺带做 Oracle 口径的
 * 校准性初检（pOk 是否真是"答对概率"），为后续"② Oracle 可靠性"铺路。
 * 输出：paper/_run/sim_selfcheck.txt
 */
const fs = require('fs');
const path = require('path');
const {answerStep, simulateSession, generateProfiles, mulberry32} = require('../app/server/core/studentSimulator');

const OUT = path.join(__dirname, '_run', 'sim_selfcheck.txt');
const L = [];
const log = (s = '') => L.push(String(s));
const bar = () => log('----------------------------------------------');

// 固定时长 fixture，使 win 仅由 rng()<pOk 决定
const CLEAN = {latencyMode: 'fixture', latencyOffsets: [8000]};

// 校准归总：以 (mastery,slip,guess) 计算真实 pOk，再统计实测反击（取真实 pOk 对比）
function empStat(mastery, slip, guess, difficulty, N, seedBase) {
  let w = 0, pOk = null;
  for (let i = 0; i < N; i++) {
    const s = answerStep({difficulty}, mastery, {slip, guess}, {...CLEAN, rng: mulberry32(seedBase + i)});
    if (i === 0) pOk = s.pOk;
    w += s.win;
  }
  return {rate: w / N, pOk};
}

const t0 = Date.now();

log('=== ① 模拟器不变式自检（数值证据） ===');
bar();

// I1: pOk 边缘一致性（mastery=0.6,slip=0.1,guess=0.25 ⇒ 真实 pOk=0.64）
const N = 200000;
const {rate, pOk} = empStat(0.6, 0.1, 0.25, 3, N, 1000);
const se = Math.sqrt(pOk * (1 - pOk) / N);
log(`[I1 边缘一致] mastery=0.6 ⇒ 真实 pOk=${pOk.toFixed(4)}  N=${N}  实测答对率=${rate.toFixed(4)}  偏差=${(rate - pOk).toFixed(4)}  (3se=${(3 * se).toFixed(4)})`);
log(`  ⇒ ${Math.abs(rate - pOk) < 3 * se ? '✓' : '✗'} 实测频率落在真实 pOk 的 3σ 带内（Oracle 口径根基因此成立）`);

// I5: 遗忘衰减
const q = [{topic: 't', difficulty: 3}, {topic: 't', difficulty: 3}];
const profNoLearn = {name: 'm', prior: 0.5, learn: 0, slip: 0.1, guess: 0.25};
const g0 = simulateSession(q, profNoLearn, {...CLEAN, rng: mulberry32(5), interAttemptGapMs: 0});
const gL = simulateSession(q, profNoLearn, {...CLEAN, rng: mulberry32(5), interAttemptGapMs: 240000});
log(`[I5 遗忘] 二次作答 pOk：间隔0=(${g0[1].pOk.toFixed(4)})  间隔240s=(${gL[1].pOk.toFixed(4)})  衰减=${(g0[1].pOk - gL[1].pOk).toFixed(4)}`);

// I6: BKT 难度耦合
const e = answerStep({difficulty: 1}, 0.5, {slip: 0.1, guess: 0.25}, {...CLEAN, rng: mulberry32(1)});
const h = answerStep({difficulty: 5}, 0.5, {slip: 0.1, guess: 0.25}, {...CLEAN, rng: mulberry32(2)});
log(`[I6 难度耦合] pOk：难度1=(${e.pOk.toFixed(4)})  难度5=(${h.pOk.toFixed(4)})  差=${(e.pOk - h.pOk).toFixed(4)}`);

// I7: regime × 权重分布（random 路径，mastery=0.5 slip=0.3 guess=0.3）
{
  const freq = {know: 0, slip: 0, guess: 0, unk: 0};
  const M = 200000;
  for (let i = 0; i < M; i++) {
    const s = answerStep({difficulty: 3}, 0.5, {slip: 0.3, guess: 0.3}, {rng: mulberry32(3000 + i), latencyMode: 'random'});
    freq[s.regime]++;
  }
  log(`[I3/7 regime] mastery=0.5,slip=0.3,guess=0.3  N=${M}`);
  for (const k of ['know', 'slip', 'guess', 'unk'])
    log(`   ${k}: ${((freq[k] / M) * 100).toFixed(2)}%  ${k === 'know' || k === 'slip' ? '(know+slip 理论 50%)' : '(guess+unk 理论 50%)'}`);
  log(`   know+slip 合计=${(((freq.know + freq.slip) / M) * 100).toFixed(2)}%`);
}

// 默认多画像（论文口径 weak/medium/strong）区分度
{
  const defQuestions = [
    {topic: 'a', difficulty: 2, question_type: 'single'},
    {topic: 'b', difficulty: 3, question_type: 'single'},
    {topic: 'c', difficulty: 4, question_type: 'single'}
  ];
  const ev = generateProfiles(defQuestions, {sessionsPerProfile: 100, seed: 7, difficultyVariance: true});
  const agg = {};
  const modelCnt = {};
  for (const x of ev) {
    agg[x.profile] = agg[x.profile] || [0, 0];
    agg[x.profile][0] += x.correct; agg[x.profile][1]++;
    modelCnt[x.model] = (modelCnt[x.model] || 0) + 1;
  }
  log('[I9 默认多画像弱<强] 平均答对率：');
  for (const k of Object.keys(agg)) log(`   ${k}: ${(agg[k][0] / agg[k][1]).toFixed(4)}  (n=${agg[k][1]})`);
  log(`   模型混合：${JSON.stringify(modelCnt)}`);
}

// Oracle 口径校准性初检：pOk 分桶，实测频率 vs 桶中心
{
  log('[校准初检(铺垫②)] 用 generateProfiles 生成一批事件，按 pOk 分桶看实测答对率是否贴着 pOk');
  const defQuestions = [
    {topic: 'a', difficulty: 2, question_type: 'single'},
    {topic: 'b', difficulty: 3, question_type: 'single'},
    {topic: 'c', difficulty: 4, question_type: 'single'}
  ];
  const evs = generateProfiles(defQuestions, {sessionsPerProfile: 200, seed: 9, difficultyVariance: true});
  const bins = {};
  for (const x of evs) {
    const b = Math.min(9, Math.floor(x.pOk * 10));
    bins[b] = bins[b] || [0, 0];
    bins[b][0] += x.correct; bins[b][1]++;
  }
  for (let b = 0; b < 10; b++) {
    if (!bins[b] || bins[b][1] === 0) continue;
    const c0 = (b + 0.5) / 10;
    log(`   pOk∈[${(b / 10).toFixed(1)},${((b + 1) / 10).toFixed(1)}): 桶中心=${c0.toFixed(2)}  实测=${(bins[b][0] / bins[b][1]).toFixed(3)}  n=${bins[b][1]}`);
  }
}

log('');
log(`（脚本耗时 ${Date.now() - t0} ms）`);
fs.writeFileSync(OUT, L.join('\n'), 'utf8');
console.log(L.join('\n'));
console.log('\n报告已写入 ' + OUT);