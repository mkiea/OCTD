/**
 * 线上数据集 ai_train 的 oracle 信息量探针（只重跑 bridge，不训练/不门禁/不部署）
 * ==============================================================================
 * 用线上同一配置(sessionsPerProfile=40/seed=42/difficultyVariance/regimeSupervision)
 * 重新生成事件并 buildTrainingSet + pOk:true，写成本地 ai_train.pok.jsonl，
 * 供 infoProbe.py --oracle 读取，得出线上合成分布的理论可预测性上界。
 * 同时打印与既有 training/data/ai_train.jsonl 的行数一致性，作为"同分布"的可信度旁证。
 */
const path = require('path');
const fs = require('fs');
const sim = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'studentSimulator'));
const bd = require(path.join(__dirname, '..', '..', 'app', 'server', 'core', 'dktDataBuilder'));
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));

const OPTS = { sessionsPerProfile: 40, numTopics: 192, maxLen: 16, seed: 42, difficultyVariance: true, regimeSupervision: true };

(async () => {
  const approved = await u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved']);
  const simOpts = { sessionsPerProfile: OPTS.sessionsPerProfile, seed: OPTS.seed, difficultyVariance: true };
  const events = sim.generateProfiles(approved, simOpts);
  const r = bd.buildTrainingSet(events, { numTopics: OPTS.numTopics, maxLen: OPTS.maxLen, regimeSupervision: true, pOk: true });
  const outPath = 'training/data/ai_train.pok.jsonl';
  fs.writeFileSync(outPath, r.rows.map((x) => JSON.stringify(x)).join('\n'), 'utf8');
  // 一致性旁证：行数与既有线上 jsonl 比较
  let oldCount = null, same = false;
  try {
    oldCount = fs.readFileSync('training/data/ai_train.jsonl', 'utf8').split('\n').filter(Boolean).length;
    same = oldCount === r.rows.length;
  } catch (e) { /* 旧文件可能不存在 */ }
  const edge = r.graph_edges ? r.graph_edges.length : 0;
  fs.writeFileSync('training/data/ai_train.pok.info.txt',
    `rows=${r.rows.length} old_ai_train=${oldCount} same_count=${same} topics=${r.topic_count} events=${r.event_count} edges=${edge}`,
    'utf8');
  console.log(`rows=${r.rows.length} old_ai_train=${oldCount} same_count=${same} topics=${r.topic_count} events=${r.event_count} -> ${outPath}`);
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });