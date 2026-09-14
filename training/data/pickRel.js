const fs = require('fs');
const t = fs.readFileSync('training/data/topics.txt', 'utf8').split(/\n/).map(s => s.trim()).filter(Boolean);
const kw = ['变法', '统一', '郡县', '焚书', '科举', '丝绸之路', '甲午', '洋务', '戊戌', '辛亥', '新文化', '五四', '抗日', '建党', '西安事变', '九一八', '抗战', '商鞅', '秦始皇', '休养', '制度', '战争', '运动', '革命', '改革', '占领', '侵略'];
const hit = t.filter(x => kw.some(k => x.includes(k)));
fs.writeFileSync('training/data/rel_topics.txt', hit.join('\n'), 'utf8');
console.log('candidate topics:', hit.length, '/ total', t.length);