const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  // 与主 p0_prereq 完全一致，仅关难度方差：difficultyVariance=false → 难度恒 3
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: false, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  const r = bd.buildTrainingSet(ev, { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm, pres: true, pOk: true });
  fs.writeFileSync('training/data/p0_prereq_nodiff.jsonl', r.rows.map(x => JSON.stringify(x)).join('\n'));
  console.log('rows=' + r.rows.length + ' topics=' + r.topic_count);
  // 抽查 diffs 是否恒 3
  const raw = fs.readFileSync('training/data/p0_prereq_nodiff.jsonl', 'utf8').split('\n');
  const hit = JSON.parse(raw[0]);
  console.log('sample diffs=', JSON.stringify(hit.diffs));
})().catch((e) => console.log('ERR', e.message, (e.stack||'').split('\n')[1]));