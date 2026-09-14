const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const present = fs.readFileSync('training/data/rel_present.txt', 'utf8').split(/\n/).map(s => s.trim()).filter(Boolean);
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true };
  const ev = sim.generateProfiles(approved, opts);
  const seen = new Set(ev.map(e => e.topic));
  const inTrace = present.filter(x => seen.has(x));
  fs.writeFileSync('training/data/rel_intrace.txt', inTrace.join('\n'), 'utf8');
  console.log('rel topics IN trace:', inTrace.length);
  console.log(JSON.stringify(inTrace));
})().catch((e) => console.log('ERR', e.message));