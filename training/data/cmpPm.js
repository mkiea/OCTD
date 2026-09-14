const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const base = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true };
  const evNo = sim.generateProfiles(approved, base);
  const evYes = sim.generateProfiles(approved, { ...base, prereqMap: pm });
  const seenNo = new Set(evNo.map(e => e.topic));
  const seenYes = new Set(evYes.map(e => e.topic));
  const out = [];
  out.push('ev length: no=' + evNo.length + ' yes=' + evYes.length);
  out.push('seen size: no=' + seenNo.size + ' yes=' + seenYes.size);
  const onlyNo = [...seenNo].filter(x => !seenYes.has(x));
  const onlyYes = [...seenYes].filter(x => !seenNo.has(x));
  out.push('only in no-prereq (' + onlyNo.length + '): ' + JSON.stringify(onlyNo.slice(0, 40)));
  out.push('only in prereq (' + onlyYes.length + '): ' + JSON.stringify(onlyYes.slice(0, 40)));
  // 检查首 20 条事件 topic
  out.push('--- first 15 events no-prereq ---');
  for (const e of evNo.slice(0, 15)) out.push(e.topic + ' @' + e.session);
  out.push('--- first 15 events prereq ---');
  for (const e of evYes.slice(0, 15)) out.push(e.topic + ' @' + e.session);
  fs.writeFileSync('training/data/pm_vs_no.txt', out.join('\n'), 'utf8');
  console.log('done');
})().catch((e) => console.log('ERR', e.message));