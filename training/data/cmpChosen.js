const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const base = { sessionsPerProfile: 40, attemptsPerTopic: 6, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 42, difficultyVariance: true, regimeSupervision: true };
  // monkeypatch 抓 chosen：通过统计每 session 出现主题
  const rg = (seed) => { let s = seed >>> 0; return () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  // 不用 monkeypatch，改为统计 qlist 每 session 的主题集合（进 simulateSession 前）
  const sizes = { no: [], yes: [] };
  // 复刻 generateProfiles 的 session 主题抽取（只到 chosen）
  function chosenSet(prereq) {
    const byTopic = {};
    for (const q of approved) { const t = String(q.topic || '').trim(); if (!t) continue; byTopic[t] = 1; }
    const topicNames = Object.keys(byTopic);
    let seedState = base.seed;
    const picks = new Set();
    const rng = rg((seedState++ + (0 * 7919)) >>> 0);
    const shuffled = [...topicNames].sort(() => rng() - 0.5);
    const maxTopics = Math.max(1, Math.floor(base.sessionMaxLen / base.attemptsPerTopic));
    const chosen = shuffled.slice(0, maxTopics);
    if (prereq) chosen.sort((a, b) => depth(a) - depth(b));
    chosen.forEach(x => picks.add(x));
    return { n: picks.size, list: [...picks] };
  }
  function depth(t, seen) { return 0; } // 简化，只测第一 session
  const a = chosenSet(false), b = chosenSet(true);
  fs.writeFileSync('training/data/chosen_cmp.txt', JSON.stringify({ no: a.n, yes: b.n, noList: a.list, yesList: b.list }, null, 0), 'utf8');
  console.log('done');
})().catch((e) => console.log('ERR', e.message));