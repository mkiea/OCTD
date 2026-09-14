/**
 * 探针④d·机制注入：生成"真实认知机制 know/slip/guess/unk 作为输入特征"探针数据
 * ============================================================================
 * 与 genP0 同配置/seed → 与 p0_prereq 逐行同序。
 * 唯一差异：buildTrainingSet opts.regimeInput=true → regimes 列写入每步真实机制编码(know=0/slip=1/guess=2/unk=3)。
 * 保留 pOk 供 oracle 计量；可比对基线/校准/机制注入在同一固定测试集上的表现。
 */
const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  const r = bd.buildTrainingSet(ev, { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm, pres: false, pOk: true, regimeInput: true });
  fs.writeFileSync('training/data/regime_probe.jsonl', r.rows.map(x => JSON.stringify(x)).join('\n'), 'utf8');
  fs.writeFileSync('training/data/regime_gen.txt',
    `rows=${r.rows.length}\n` +
    `regimes 列 = 每步真实认知机制编码(know=0/slip=1/guess=2/unk=3)，pOk 保留\n` +
    `同序来源：与 p0_prereq 同配置生成 → 可复用 p0_prereq.split.json\n`, 'utf8');
  console.log('done rows=', r.rows.length);
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });