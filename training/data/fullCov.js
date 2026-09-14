const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  // 给 x 一个 profile，单会话，sessionMaxLen 拉满到覆盖全部主题
  const out = [];
  out.push('approved rows: ' + approved.length);
  out.push('prereqMap dst sample(src/dst都要在数据里): ' + JSON.stringify(
    Object.keys(pm).slice(0, 5).map(k => [k, pm[k].needs])
  ));
  // 关键：让 chosen 覆盖所有主题 -> sessionMaxLen 设大
  const opts = { profiles: [{ name: 'm', prior: 0.4, learn: 0.1, slip: 0.1, guess: 0.2, a: 1.2 }], sessionsPerProfile: 1, attemptsPerTopic: 2, sessionMaxLen: 100000, interAttemptGapMs: 1000, seed: 42, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  const seen = new Set(ev.map(e => e.topic));
  out.push('trace topics (covered all?): ' + seen.size);
  const miss = Object.keys(pm).filter(k => !seen.has(k));
  out.push('prereqMap dst missing: ' + JSON.stringify(miss));
  fs.writeFileSync('training/data/fullcov.txt', out.join('\n'), 'utf8');
  console.log('done', ev.length);
})().catch((e) => console.log('ERR', e.message));