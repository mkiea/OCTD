/**
 * 探针④·隐态天花板：生成"真掌握度注入"探针数据（复用 genP0 同配置/seed → 与 p0_prereq 逐行同序）
 * ============================================================================
 * 唯一差异：buildTrainingSet opts.mtrue=true → 把"作答前真掌握度"覆写入 pres 通道（供 --use-pres 喂入模型）。
 * 其余生成参数与 genP0 完全一致，故 mtrue_probe.jsonl 的 rows/顺序 与 p0_prereq.jsonl 一一对应，
 * 可复用 training/data/p0_prereq.split.json 的行级 test 划分，保证与可观察基线在同一测试集上对比。
 */
const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  // 与 genP0.js 完全一致：(sessionsPerProfile=40/attemptsPerTopic=6/sessionMaxLen=192/interAttemptGapMs=45000/seed=42/difficultyVariance/regimeSupervision/prereqMap)
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  // mtrue=true → pres 通道覆写为"作答前真掌握度"；保留 pOk 供 oracle 计量
  const r = bd.buildTrainingSet(ev, { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm, pres: false, pOk: true, mtrue: true });
  fs.writeFileSync('training/data/mtrue_probe.jsonl', r.rows.map(x => JSON.stringify(x)).join('\n'), 'utf8');
  fs.writeFileSync('training/data/mtrue_gen.txt',
    `rows=${r.rows.length}\n` +
    `pres 通道 = 作答前真掌握度(mtrue)（覆写），pOk 保留\n` +
    `同序来源：与 p0_prereq 同配置生成 → 可复用 p0_prereq.split.json\n`, 'utf8');
  console.log('done rows=', r.rows.length);
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });