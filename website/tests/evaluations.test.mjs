import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { bestModelIds, comparison, percent, rankModels, categoryScores, filterExamples, filterItems } from '../evaluations.mjs';

const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const data = JSON.parse(read('../assets/evaluations/results.json'));
const examples = JSON.parse(read('../assets/evaluations/public-examples.json'));
const models = data.models;
const ids = models.map(m => m.id).sort();

test('vision winners exclude unavailable results and preserve ties', () => {
  assert.deepEqual(bestModelIds({ a: 0.9, b: 0.7, jev: null }), ['a']);
  assert.deepEqual(bestModelIds({ a: 0.9, b: 0.9, jev: null }), ['a', 'b']);
  assert.deepEqual(bestModelIds({ a: 0, jev: null }), ['a']);
  assert.deepEqual(bestModelIds({ jev: null }), []);
});

test('comparison highlights strict winners, not the baseline or ties', () => {
  assert.equal(comparison(214 / 231, 200 / 231), 'win');
  assert.equal(comparison(199 / 231, 200 / 231), 'loss');
  assert.equal(comparison(1, 1), 'tie');
  assert.equal(comparison(0.3, 0.1 + 0.2), 'tie');
  assert.equal(comparison(200 / 231, 200 / 231, true), 'baseline');
  assert.equal(comparison(1, 0), 'win');
  assert.equal(comparison(0, 1), 'loss');
  assert.equal(comparison(0, 0), 'tie');
});

test('headline matches the agreed table, with the published Jev baseline', () => {
  assert.deepEqual(models.map(m => [m.id, m.publicCorrect, percent(m.decisionScore)]), [
    ['qwen27b', 214, '89.87%'], ['gemma-moe', 207, '88.77%'], ['qwen-moe', 204, '88.60%'],
    ['jev', 200, '87.15%'], ['gemma12b', 199, '83.37%'], ['qwen4b', 178, '78.71%'],
  ]);
  assert.equal(percent(200 / 231), '86.58%');
  assert.equal(data.publishedReference.correct, 200);
  assert.deepEqual(models.map(m => m.policy), ['examples_binary', 'strict_mix_repeat2', 'repeat_state', 'Native', 'strict_mix_repeat2', 'strict_mix_repeat2']);
});

test('each public question has authentic gold and six outcomes; no invented Jev label', () => {
  assert.equal(examples.length, 231);
  assert.equal(new Set(examples.map(e => e.id)).size, 231);
  for (const e of examples) {
    assert.deepEqual(Object.keys(e.answers).sort(), ids);
    assert.ok(e.question.instructions);
    assert.ok(['choice', 'score', 'noul'].includes(e.question.type));
    assert.notEqual(e.expected, null);
    assert.equal(e.answers.jev.predicted, null);
    for (const [model, answer] of Object.entries(e.answers)) {
      assert.equal(typeof answer.correct, 'boolean');
      if (model !== 'jev') assert.equal(String(answer.predicted) === String(e.expected), answer.correct);
    }
  }
  for (const model of models) assert.equal(examples.filter(e => e.answers[model.id].correct).length, model.publicCorrect);
});

test('all public breakdowns recompute from individual outcomes', () => {
  for (const [dimension, groups] of Object.entries(data.publicBreakdowns)) {
    assert.equal(groups.reduce((sum, g) => sum + g.rows, 0), 231);
    for (const group of groups) {
      const members = examples.filter(e => (dimension === 'primitive' ? e.question.type : e[dimension]) === group.name);
      assert.equal(members.length, group.rows);
      for (const model of models) assert.equal(members.filter(e => e.answers[model.id].correct).length, group.scores[model.id]);
    }
  }
  assert.deepEqual(Object.fromEntries(data.publicBreakdowns.tier.map(g => [g.name, g.rows])), { easy: 48, hard: 111, original: 72 });
  assert.deepEqual(data.publicBreakdowns.tier.map(g => g.scores.jev), [48, 81, 71]);
});

test('decision headline is the mean of 26 matched items, not pooled rows', () => {
  assert.equal(data.decisionItems.length, 26);
  assert.equal(data.decisionRows, 21364);
  assert.equal(data.decisionQuestions, 33099);
  assert.equal(data.decisionItems.reduce((sum, item) => sum + item.rows, 0), data.decisionRows);
  assert.equal(data.decisionItems.reduce((sum, item) => sum + item.questions, 0), data.decisionQuestions);
  assert.equal(new Set(data.decisionItems.map(item => item.id)).size, 26);
  for (const item of data.decisionItems) {
    assert.deepEqual(Object.keys(item.scores).sort(), ids);
    assert.ok(item.rows > 0 && item.description && item.scoring);
    assert.ok(item.example.id && item.example.questions && item.example.gold);
    assert.ok(item.suites.includes(item.example.suite));
    for (const value of Object.values(item.scores)) assert.ok(Number.isFinite(value) && value >= 0 && value <= 1);
  }
  for (const model of models) {
    const mean = data.decisionItems.reduce((sum, item) => sum + item.scores[model.id], 0) / 26;
    assert.ok(Math.abs(mean - model.decisionScore) < 1e-12);
  }
  const categories = categoryScores(data.decisionItems, models);
  assert.equal(categories.length, 6);
  assert.equal(categories.reduce((sum, c) => sum + c.rows, 0), 26);
  for (const model of models) {
    assert.ok(Math.abs(categories.reduce((sum, c) => sum + c.scores[model.id] * c.rows, 0) / 26 - model.decisionScore) < 1e-12);
  }
});

test('vision is seven matched accuracy configurations with no invented Jev score', () => {
  assert.equal(data.visionItems.length, 7);
  assert.equal(data.visionRows, 63372);
  assert.equal(data.visionItems.reduce((sum, item) => sum + item.rows, 0), 63372);
  assert.deepEqual(models.map(m => m.visionScore == null ? null : percent(m.visionScore)), ['86.52%', '86.14%', '88.29%', null, '83.00%', '85.62%']);
  for (const model of models) {
    if (model.id === 'jev') { assert.equal(model.visionScore, null); continue; }
    const mean = data.visionItems.reduce((sum, item) => sum + item.scores[model.id], 0) / 7;
    assert.ok(Math.abs(mean - model.visionScore) < 1e-12);
  }
  for (const item of data.visionItems) {
    assert.equal(item.scores.jev, null);
    assert.equal(item.nativeScores.jev, null);
    assert.deepEqual(Object.keys(item.scores).sort(), ids);
    const bytes = readFileSync(new URL(`../${item.example.image}`, import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), item.example.imageSha256);
    assert.ok(item.example.question && item.example.gold != null);
    for (const model of models.filter(m => m.id !== 'jev')) {
      assert.ok(item.scores[model.id] >= 0 && item.scores[model.id] <= 1);
    }
  }
  const pope = data.visionItems.filter(item => item.project === 'POPE');
  assert.deepEqual(pope.map(item => item.example.id), ['2', '8', '14']);
  assert.equal(new Set(pope.map(item => item.example.imageSha256)).size, 3);
  assert.equal(new Set(pope.map(item => item.example.question)).size, 3);
  assert.ok(pope.every(item => item.example.gold === 'no'));
  const cifar = data.visionItems.find(item => item.project === 'CIFAR-10');
  assert.equal(cifar.example.id, 'image-000010');
  assert.equal(cifar.example.gold, 'airplane');
  assert.match(cifar.example.selection, /visual clarity/);
  const mme = data.visionItems.find(item => item.project === 'MME');
  assert.equal(mme.nativeMetric, 'mme_score');
  assert.ok(mme.nativeScores.qwen27b > 1000);
  assert.notEqual(mme.scores.qwen27b, mme.nativeScores.qwen27b / 2000);
  const ranked = rankModels(models, 'visionScore');
  assert.equal(ranked[0].id, 'qwen-moe');
  assert.equal(ranked.at(-1).id, 'jev');
  assert.equal(ranked.at(-1).rank, null);
});

test('rankings are deterministic, handle ties, and do not mutate input', () => {
  assert.deepEqual(rankModels(models, 'decisionScore').map(m => m.rank), [1, 2, 3, 4, 5, 6]);
  const tied = [{ publicCorrect: 2 }, { publicCorrect: 3 }, { publicCorrect: 3 }];
  assert.deepEqual(rankModels(tied, 'publicCorrect').map(m => m.rank), [1, 1, 3]);
  assert.equal(tied[0].publicCorrect, 2);
  assert.throws(() => rankModels(models, 'unknown'));
});

test('question filters handle combined filters and empty selections', () => {
  assert.equal(filterExamples(examples, 'all', 'all', 'all').length, 231);
  assert.equal(filterExamples(examples, 'original', 'all', 'all').length, 72);
  assert.equal(filterExamples(examples, 'easy', 'score', 'all').length, 0);
  const misses = filterExamples(examples, 'hard', 'noul', 'miss');
  assert.ok(misses.length > 0);
  assert.ok(misses.every(e => e.tier === 'hard' && e.question.type === 'noul' && Object.values(e.answers).some(a => !a.correct)));
});

test('decision search and domain filters do not change the snapshot', () => {
  assert.equal(filterItems(data.decisionItems, 'all', ' semIF ').length, 4);
  assert.equal(filterItems(data.decisionItems, 'legal', '').length, 3);
  assert.equal(filterItems(data.decisionItems, 'legal', 'SemIf').length, 0);
  assert.equal(data.decisionItems.length, 26);
});

test('source manifest includes the canonical evaluator and selected predictions', () => {
  assert.ok(Object.keys(data.sources).some(p => p.startsWith('simple-jev-eval/eval/suites/')));
  assert.equal(Object.keys(data.sources).filter(p => p.endsWith('/predictions.jsonl')).length, 5);
  assert.ok(data.sources['reports/FULL_EVAL_FINAL_COMPARISON.json']);
  for (const hash of Object.values(data.sources)) assert.match(hash, /^[a-f0-9]{64}$/);
  assert.match(read('../assets/evaluations/LICENSE-jevbench.txt'), /MIT License/);
});

test('page has accessible disclosures, prompt column, caveats, sources and no unsafe HTML sink', () => {
  const html = read('../evaluations.html');
  for (const id of ['public-set', 'decision-set', 'vision-set', 'methodology']) assert.ok(html.includes(`<details id="${id}"`));
  assert.match(html, /Preferred prompt format/);
  const header = html.split('<header')[1].split('</header>')[0];
  assert.match(header, /assets\/simple-jev\.png/);
  assert.match(header, /assets\/featherless_logo_dark\.svg/);
  assert.match(header, /href="how-it-works.html"/);
  assert.match(header, /href="index.html#rfdt"/);
  assert.doesNotMatch(read('../evaluations.mjs'), /Winner · vision mean|winner-badge/);
  assert.match(html, /not held-out accuracy/);
  assert.doesNotMatch(html.split('<details id="public-set"')[0], /199\/231/);
  const publicSection = html.split('<details id="public-set"')[1].split('<details id="decision-set"')[0];
  assert.match(publicSection, /199\/231 \(86\.15%\)/);
  assert.match(publicSection, /we could not replicate the published/);
  assert.match(html, /Jev’s baseline is <strong>200\/231 \(86\.58%\)/);
  assert.match(html, /534-decision/);
  assert.match(html, /≥0.5/);
  assert.match(html, /<noscript>/);
  assert.doesNotMatch(read('../evaluations.mjs'), /innerHTML|outerHTML|insertAdjacentHTML|localStorage/);
  for (const file of ['index', 'playground', 'docs', 'demos', 'how-it-works']) assert.ok(read(`../${file}.html`).includes('href="evaluations.html"'));
  for (const [, path] of html.matchAll(/(?:src|href)="([^"#][^"]*)"/g)) {
    if (/^https:/.test(path)) continue;
    assert.ok(existsSync(new URL(`../${path.split(/[?#]/)[0]}`, import.meta.url)), path);
  }
});
