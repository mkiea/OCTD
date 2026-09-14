const path = require('path');
const fs = require('fs');
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
// 造一小段含先修依赖的人工 trace：先有"商鞅"，后有"商鞅变法"等
const pm = JSON.parse(fs.readFileSync('training/data/prereqMap.json', 'utf8'));
console.log('prereqMap keys sample:', Object.keys(pm).slice(0, 6));
const events = [];
// 构造 session1: 商鞅(先修) -> 商鞅变法
for (let i = 0; i < 8; i++) {
  events.push({ topic: '商鞅', correct: 1, difficulty: 3, ts: i * 1000, session: 's1', regime: 'know' });
  events.push({ topic: '商鞅变法', correct: i < 5 ? 1 : 0, difficulty: 3, ts: i * 1000 + 500, session: 's1', regime: 'know' });
}
const r = bd.buildTrainingSet(events, { numTopics: 192, maxLen: 16, regimeSupervision: true, prereqMap: pm });
console.log('rows', r.rows.length, 'distinctTopics', r.topic_count);
console.log('edges', r.graph_edges.length, JSON.stringify(r.graph_edges.slice(0, 10)));