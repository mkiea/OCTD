/**
 * 加熵策略 oracle 信息增益评估（投钱前检查）
 * =====================================================================
 * 同一生成配置（sessionsPerProfile=40/attemptsPerTopic=6/sessionMaxLen=192/seed=42/prereqMap）下，
 * 对比"同主题连答(基线)" vs "跨主题交错作答(加熵候选)" 的理论可预测性上界(oracle)。
 * 判定：交错 oracle > 基线上界 ⇒ 加熵带来确定性信息增益(数据内有更多可预测可提取信号)；
 *       交错 oracle ≤ 基线上界 ⇒ 加熵不明显提高数据内可预测性(增益若在则来自贴合真实，非合成内)。
 */
const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));

function oracle(rows) {
  const p = [], y = [];
  for (const r of rows) {
    if (!r.pOk) return null;
    for (let t = 0; t < r.pOk.length; t++) {
      if (Number.isFinite(r.pOk[t])) { p.push(r.pOk[t]); y.push(r.corrects[t]); }
    }
  }
  const n = y.length, pos = [], neg = [];
  for (let i = 0; i < n; i++) (y[i] === 1 ? pos : neg).push(p[i]);
  if (!pos.length || !neg.length) return null;
  const all = [...pos, ...neg].map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  let rsum = 0;
  all.forEach(([v, idx], rank) => { if (idx < pos.length) rsum += rank + 1; });
  const auc = rsum / (pos.length * neg.length) - (pos.length + 1) / (2 * neg.length);
  return { auc: Math.max(0, Math.min(1, auc)), pos: pos.length, neg: neg.length };
}

(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, prereqMap: pm };
  const bdOpts = { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm, pres: true, pOk: true };

  const evBase = sim.generateProfiles(approved, opts);
  const evInter = sim.generateProfiles(approved, { ...opts, interleave: true });
  const rBase = bd.buildTrainingSet(evBase, bdOpts);
  const rInter = bd.buildTrainingSet(evInter, bdOpts);

  fs.writeFileSync('training/data/entropy_base.jsonl', rBase.rows.map(x => JSON.stringify(x)).join('\n'), 'utf8');
  fs.writeFileSync('training/data/entropy_inter.jsonl', rInter.rows.map(x => JSON.stringify(x)).join('\n'), 'utf8');

  const oBase = oracle(rBase.rows), oInter = oracle(rInter.rows);
  const delta = oBase && oInter ? oInter.auc - oBase.auc : null;
  console.log(`baseline(连答)  oracle=${oBase ? oBase.auc.toFixed(4) : 'n/a'}  pos=${oBase ? oBase.pos : 0}+neg=${oBase ? oBase.neg : 0}  rows=${rBase.rows.length}`);
  console.log(`interleave(交错) oracle=${oInter ? oInter.auc.toFixed(4) : 'n/a'}  pos=${oInter ? oInter.pos : 0}+neg=${oInter ? oInter.neg : 0}  rows=${rInter.rows.length}`);
  console.log(`delta=${delta == null ? 'n/a' : delta.toFixed(4)}`);
  fs.writeFileSync('training/data/entropy_report.txt',
    `base_oracle=${oBase ? oBase.auc.toFixed(4) : 0}\ninter_oracle=${oInter ? oInter.auc.toFixed(4) : 0}\ndelta=${delta == null ? 'n/a' : delta.toFixed(4)}\nbase_rows=${rBase.rows.length}\ninter_rows=${rInter.rows.length}\n`, 'utf8');
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });