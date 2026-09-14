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
  out.push('distinct topics in trace: ' + seen.size);
  out.push('--- prereqMap dst keys: in trace? ---');
  for (const k of Object.keys(pm)) out.push((seen.has(k) ? 'YES ' : 'no  ') + k);
  out.push('--- sample trace topics (first 60) ---');
  out.push([...seen].slice(0, 60).join(' | '));
  fs.writeFileSync('training/data/trace_dump.txt', out.join('\n'), 'utf8');
  console.log('done', ev.length);
})().catch((e) => console.log('ERR', e.message));