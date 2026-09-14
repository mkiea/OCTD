const fs = require('fs');
const path = require('path');
const u = require(path.join(__dirname, '..', '..', 'app', 'server', 'util'));
const rel = fs.readFileSync('training/data/rel_topics.txt', 'utf8').split(/\n/).map(s => s.trim()).filter(Boolean);
(u.connectMysql('SELECT * FROM ai_training_bank WHERE status=?', ['approved'])
  .then((rows) => {
    const dbTopics = new Set(rows.map(r => String(r.topic)));
    const present = rel.filter(x => dbTopics.has(x));
    const absent = rel.filter(x => !dbTopics.has(x));
    fs.writeFileSync('training/data/rel_present.txt', present.join('\n'), 'utf8');
    fs.writeFileSync('training/data/rel_absent.txt', absent.join('\n'), 'utf8');
    console.log('approved total rows', rows.length, '| rel present in DB', present.length, '/', rel.length);
    console.log('absent:', JSON.stringify(absent));
  }).catch((e) => console.log('ERR', e.message)));