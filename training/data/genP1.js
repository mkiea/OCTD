const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
// 第 2 分布：与 p0_prereq 同源题库(approved bank=3392)，但改变生成参数构成不同分布实现：
//   seed 42→7、attemptsPerTopic 6→8（每知识点连续作答更长，改变逐主题转移结构）、
//   sessionsPerProfile 40→60（更多会话覆盖）。difficultyVariance/regime 保持。
(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const opts = { sessionsPerProfile: 60, attemptsPerTopic: 8, sessionMaxLen: 192, interAttemptGapMs: 45000, seed: 7, difficultyVariance: true, regimeSupervision: true, prereqMap: pm };
  const ev = sim.generateProfiles(approved, opts);
  const r = bd.buildTrainingSet(ev, { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm, pres: true, pOk: true });
  fs.writeFileSync('training/data/p1_prereq.jsonl', r.rows.map(x => JSON.stringify(x)).join('\n'));
  fs.writeFileSync('training/data/p1_prereq.edges.json', JSON.stringify(r.graph_edges));
  const out = [];
  out.push('rows: p1=' + r.rows.length);
  out.push('graph_edges: ' + r.graph_edges.length);
  out.push('distinct topics: ' + r.topic_count);
  fs.writeFileSync('training/data/p1_gen.txt', out.join('\n'), 'utf8');
  console.log('done p1');
})().catch((e) => console.log('ERR', e.message));