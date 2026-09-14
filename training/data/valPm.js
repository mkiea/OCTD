const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  const seen = new Set(ev.map(e => e.topic));
  const out = [];
  let miss = [];
  for (const k of Object.keys(pm)) {
    if (!seen.has(k)) miss.push('dst:' + k);
    for (const n of (pm[k].needs || [])) if (!seen.has(n)) miss.push('need:' + n);
  }
  out.push('prereqMap entries: ' + Object.keys(pm).length);
  out.push('missing names in trace: ' + (miss.length ? JSON.stringify(miss) : 'NONE'));
  fs.writeFileSync('training/data/pm_validate.txt', out.join('\n'), 'utf8');
  console.log('done');
})().catch((e) => console.log('ERR', e.message));