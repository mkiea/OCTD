const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const byTopic = {};
  for (const q of approved) { const t = String(q.topic || '').trim(); if (t) byTopic[t] = (byTopic[t] || []).push(q) || byTopic[t]; }
  const topicNames = Object.keys(byTopic);
  const opts = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  // 对每个 profile/sess 复刻抽取，统计每个主题被选中的 session 数
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const profiles = op => (Array.isArray(op.profiles) && op.profiles.length ? op.profiles : [{name:'x'}]);
  // 只测单个 profile x，40 sessions
  let seedState = opts.seed;
  const pickYes = new Set(), pickNo = new Set();
  for (let s = 0; s < opts.sessionsPerProfile; s++) {
    const rngYes = mulberry32((seedState++ + (s * 7919)) >>> 0);
    let sh = [...topicNames].sort(() => rngYes() - 0.5);
    let ch = sh.slice(0, Math.floor(opts.sessionMaxLen / opts.attemptsPerTopic));
    ch.forEach(t => pickYes.add(t));
  }
  seedState = opts.seed;
  for (let s = 0; s < opts.sessionsPerProfile; s++) {
    const rngNo = mulberry32((seedState++ + (s * 7919)) >>> 0);
    let sh = [...topicNames].sort(() => rngNo() - 0.5);
    let ch = sh.slice(0, Math.floor(opts.sessionMaxLen / opts.attemptsPerTopic));
    ch.forEach(t => pickNo.add(t));
  }
  fs.writeFileSync('training/data/pick_diag.txt', JSON.stringify({
    totalTopics: topicNames.length,
    no: { size: pickNo.size },
    yes: { size: pickYes.size },
    onlyNo: [...pickNo].filter(x => !pickYes.has(x)),
    onlyYes: [...pickYes].filter(x => !pickNo.has(x))
  }, null, 0), 'utf8');
  console.log('done');
})().catch((e) => console.log('ERR', e.message));