// Frozen data only: no model requests, credentials, storage, or executable dataset content.
export const percent = (value) => `${(value * 100).toFixed(2)}%`;
export function bestModelIds(scores) {
  const measured = Object.entries(scores).filter(([, value]) => Number.isFinite(value));
  const best = Math.max(...measured.map(([, value]) => value));
  return measured.filter(([, value]) => Math.abs(value - best) < 1e-12).map(([id]) => id);
}
export function comparison(value, baseline, isBaseline = false) {
  if (isBaseline) return 'baseline';
  if (Math.abs(value - baseline) < 1e-12) return 'tie';
  return value > baseline ? 'win' : 'loss';
}
function markComparison(cell, value, baseline, isBaseline, showDelta = true) {
  const result = comparison(value, baseline, isBaseline);
  cell.classList.add(`comparison-${result}`);
  const labels = { baseline: 'Jev baseline', win: 'Won vs Jev', loss: 'Lost vs Jev', tie: 'Tied with Jev' };
  const delta = (value - baseline) * 100;
  const suffix = showDelta && ['win', 'loss'].includes(result) ? ` · ${delta > 0 ? '+' : ''}${delta.toFixed(2)} pp` : '';
  cell.append(node('small', labels[result] + suffix, 'comparison-label'));
}
export function rankModels(models, key) {
  if (!['publicCorrect', 'decisionScore', 'visionScore'].includes(key)) throw new Error('Unknown ranking');
  const sorted = [...models].sort((a, b) => (b[key] ?? -Infinity) - (a[key] ?? -Infinity));
  return sorted.map(model => ({ ...model, rank: model[key] == null ? null : sorted.findIndex(m => m[key] === model[key]) + 1 }));
}
export function categoryScores(items, models) {
  return [...new Set(items.map(item => item.category))].sort().map(name => {
    const group = items.filter(item => item.category === name);
    return { name, rows: group.length, scores: Object.fromEntries(models.map(model => [model.id,
      group.reduce((sum, item) => sum + item.scores[model.id], 0) / group.length])) };
  });
}
export function filterExamples(examples, tier, type, outcome) {
  return examples.filter(row => (tier === 'all' || row.tier === tier)
    && (type === 'all' || row.question.type === type)
    && (outcome === 'all' || Object.values(row.answers).some(answer => !answer.correct)));
}
export function filterItems(items, category, query) {
  const term = query.trim().toLowerCase();
  return items.filter(item => (category === 'all' || item.category === category)
    && `${item.project} ${item.configuration} ${item.category} ${item.description}`.toLowerCase().includes(term));
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
function itemToggle() {
  const hint = node('span', null, 'expand-mark item-toggle');
  hint.append(node('span', 'View scores & example', 'when-closed'), node('span', 'Hide scores & example', 'when-open'));
  const chevron = node('span', '⌄', 'disclosure-chevron'); chevron.setAttribute('aria-hidden', 'true'); hint.append(chevron);
  return hint;
}
function visionScoreCell(modelId, scores, format = percent) {
  const value = scores[modelId];
  const winners = bestModelIds(scores);
  const winner = winners.includes(modelId);
  const cell = node('td', value == null ? '—' : format(value), value == null ? 'comparison-baseline' : winner ? 'vision-winner' : 'vision-score');
  if (winner) cell.setAttribute('aria-label', `${format(value)}; ${winners.length > 1 ? 'joint highest' : 'highest'} measured score`);
  if (value == null) cell.append(node('small', 'Not evaluated'));
  return cell;
}
function pretty(value) { return typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
function pre(value) {
  const element = node('pre', pretty(value));
  element.tabIndex = 0;
  return element;
}
function tableHeader(table, headings) {
  const head = node('thead'); const row = node('tr');
  for (const label of headings) { const cell = node('th', label); cell.scope = 'col'; row.append(cell); }
  head.append(row); table.append(head);
  const body = node('tbody'); table.append(body); return body;
}
function rowHeader(text) { const th = node('th', text); th.scope = 'row'; return th; }
function scrollTable(table, label) {
  const wrap = node('div', null, 'eval-scroll'); wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', label); wrap.append(table); return wrap;
}
function contextBlock(state) {
  const details = node('details'); details.append(node('summary', 'Read full source context'), pre(state ?? '(No shared context)'));
  return details;
}
function questionBlock(questions, gold) {
  const box = node('div', null, 'eval-example');
  box.append(node('h4', 'Question / permitted answers'), pre(questions), node('h4', 'Dataset gold answer (not a model response)'), pre(gold));
  return box;
}
async function fetchJSON(path) {
  const response = await fetch(path, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
function showRetry(container, message, action) {
  container.replaceChildren(node('span', `${message} `));
  const button = node('button', 'Retry', 'retry-button'); button.type = 'button'; button.addEventListener('click', action);
  container.append(button);
}

async function init() {
  const $ = id => document.getElementById(id);
  let data;
  try {
    data = await fetchJSON('assets/evaluations/results.json');
    if (data.schemaVersion !== 1 || data.models.length !== 6 || data.decisionItems.length !== 26) throw new Error('Invalid snapshot');
    $('load-status').textContent = '';
  } catch {
    showRetry($('load-status'), 'Could not load the evaluation snapshot. Serve this page over HTTP/HTTPS and try again.', init);
    return;
  }
  const models = data.models;
  function leaderboard(key) {
    const body = $('leaderboard'); body.replaceChildren();
    const baseline = models.find(model => model.id === 'jev');
    const bestVision = Math.max(...models.map(model => model.visionScore ?? -Infinity));
    for (const model of rankModels(models, key)) {
      const row = node('tr', null, model.id === 'jev' ? 'reference-row' : key === 'visionScore'
        ? (model.visionScore === bestVision ? 'vision-leader-row' : '') : model[key] > baseline[key] ? 'winner-row' : '');
      const name = rowHeader(model.name);
      const policy = node('td'); policy.append(node('code', model.policy));
      row.append(node('td', model.rank ?? '—'), name, policy);
      for (const [value, reference, display, note] of [[model.publicCorrect / 231, baseline.publicCorrect / 231, `${model.publicCorrect} — ${percent(model.publicCorrect / 231)}`, 'of 231 decisions'],
        [model.decisionScore, baseline.decisionScore, percent(model.decisionScore), 'Mean of 26 items']]) {
        const winning = model.id !== 'jev' && value > reference;
        const cell = node('td', null, winning ? 'winning-score' : ''); const track = node('div', null, 'score-track'); track.setAttribute('aria-hidden', 'true');
        const fill = node('div', null, 'score-fill'); fill.style.width = `${value * 100}%`; track.append(fill);
        cell.append(node('span', display, 'score-number'), track, node('small', note));
        markComparison(cell, value, reference, model.id === 'jev');
        row.append(cell);
      }
      const visionCell = node('td', null, model.visionScore == null ? '' : 'vision-score');
      if (model.visionScore == null) {
        visionCell.append(node('span', '—', 'score-number'), node('small', 'Not evaluated'));
      } else {
        const track = node('div', null, 'score-track'); track.setAttribute('aria-hidden', 'true');
        const fill = node('div', null, 'score-fill'); fill.style.width = `${model.visionScore * 100}%`; track.append(fill);
        visionCell.append(node('span', percent(model.visionScore), 'score-number'), track, node('small', 'Mean accuracy · 7 configs'));
        if (model.visionScore === bestVision) {
          visionCell.classList.add('vision-winner');
          visionCell.setAttribute('aria-label', `${percent(model.visionScore)}; highest measured vision mean`);
        }
      }
      row.append(visionCell); body.append(row);
    }
    for (const button of document.querySelectorAll('[data-sort]')) {
      const selected = button.dataset.sort === key;
      button.parentElement.setAttribute('aria-sort', selected ? 'descending' : 'none');
      button.firstChild.textContent = ({ publicCorrect: 'JevBench public set', decisionScore: 'Decision benchmarks', visionScore: 'Vision benchmarks' }[button.dataset.sort]) + (selected ? ' ↓' : '');
    }
  }
  leaderboard('publicCorrect');
  document.querySelectorAll('[data-sort]').forEach(button => button.addEventListener('click', () => leaderboard(button.dataset.sort)));

  function breakdown(id, groups, publicSet) {
    const table = $(id); table.replaceChildren(); table.className = 'breakdown-table';
    const comparisonModels = [...models].sort((a, b) => Number(b.id === 'jev') - Number(a.id === 'jev'));
    const body = tableHeader(table, [publicSet ? 'Group / decisions' : 'Domain / items', ...comparisonModels.map(m => m.id === 'jev' ? `${m.name} · baseline` : m.name)]);
    for (const group of groups) {
      const row = node('tr'); const name = rowHeader(group.name.replaceAll('_', ' '));
      name.append(node('small', `${group.rows} ${publicSet ? 'decisions' : 'items'}`)); row.append(name);
      for (const model of comparisonModels) {
        const value = group.scores[model.id];
        const denominator = publicSet ? group.rows : 1;
        const cell = node('td', percent(value / denominator));
        if (publicSet) cell.append(node('small', `${value} / ${group.rows}`));
        markComparison(cell, value / denominator, group.scores.jev / denominator, model.id === 'jev');
        row.append(cell);
      }
      body.append(row);
    }
  }
  const publicBreakdown = () => breakdown('public-breakdown', data.publicBreakdowns[$('public-dimension').value], true);
  publicBreakdown(); $('public-dimension').addEventListener('change', publicBreakdown);
  const categories = categoryScores(data.decisionItems, models);
  breakdown('category-breakdown', categories, false);
  for (const group of categories) { const option = node('option', group.name); option.value = group.name; $('decision-category').append(option); }

  function renderDecisionItems() {
    const items = filterItems(data.decisionItems, $('decision-category').value, $('decision-search').value);
    $('decision-status').textContent = `${items.length} of 26 items shown. Filters do not change the headline score.`;
    $('decision-items').replaceChildren();
    for (const item of items) {
      const details = node('details', null, 'item-detail'); const summary = node('summary', `${item.project} · ${item.configuration}`);
      summary.append(node('small', `${item.category} · ${item.questions.toLocaleString()} scored questions · ${item.rows.toLocaleString()} input rows · weight 1/26`), itemToggle()); details.append(summary);
      const body = node('div', null, 'item-body');
      body.append(node('p', item.description), node('p', item.scoring));
      if (item.scopeNote) body.append(node('p', item.scopeNote));
      const table = node('table'); const scores = tableHeader(table, ['Model', 'Item accuracy']);
      for (const model of [...models].sort((a,b) => Number(b.id === 'jev') - Number(a.id === 'jev') || item.scores[b.id] - item.scores[a.id])) {
        const row = node('tr', null, model.id === 'jev' ? 'reference-row' : comparison(item.scores[model.id], item.scores.jev) === 'win' ? 'winner-row' : '');
        const score = node('td', percent(item.scores[model.id]));
        markComparison(score, item.scores[model.id], item.scores.jev, model.id === 'jev');
        row.append(rowHeader(model.id === 'jev' ? `${model.name} · baseline` : model.name), score); scores.append(row);
      }
      body.append(scrollTable(table, `${item.project} item scores`), node('h4', 'An actual evaluation example'),
        node('p', `Case ${item.example.id} · ${item.example.suite}. A short case from the first constituent suite, not a representative sample of model quality.`),
        contextBlock(item.example.state), questionBlock(item.example.questions, item.example.gold));
      const provenance = node('details'); provenance.append(node('summary', 'Source and constituent suites'), pre({ suites: item.suites, source: item.source }));
      body.append(provenance); details.append(body); $('decision-items').append(details);
    }
  }
  renderDecisionItems();
  $('decision-search').addEventListener('input', renderDecisionItems);
  $('decision-category').addEventListener('change', renderDecisionItems);

  const visionModels = rankModels(models, 'visionScore'); // Unmeasured Jev is last, never treated as zero.
  const visionTable = $('vision-breakdown'); visionTable.className = 'breakdown-table vision-matrix';
  const visionBody = tableHeader(visionTable, ['Benchmark / questions', ...visionModels.map(m => m.name)]);
  const visionItems = [...data.visionItems].sort((a, b) => Number(a.id === 'vision-cifar10') - Number(b.id === 'vision-cifar10'));
  for (const item of visionItems) {
    const row = node('tr'); const label = rowHeader(`${item.project} · ${item.configuration}`);
    label.append(node('small', `${item.rows.toLocaleString()} questions`)); row.append(label);
    for (const model of visionModels) row.append(visionScoreCell(model.id, item.scores));
    visionBody.append(row);
    const details = node('details', null, 'item-detail');
    details.dataset.suite = item.id;
    const summary = node('summary', `${item.project} · ${item.configuration}`);
    summary.append(node('small', `${item.rows.toLocaleString()} questions · weight 1/7`), itemToggle()); details.append(summary);
    const body = node('div', null, 'item-body'); body.append(node('p', item.description));
    const table = node('table');
    const metricLabel = item.nativeMetric === 'mme_score' ? 'Native MME / 2,000' : item.nativeMetric === 'f1' ? 'Native F1' : 'Native accuracy';
    const hasDistinctMetric = item.nativeMetric !== 'accuracy';
    const scores = tableHeader(table, ['Model', 'Question accuracy', ...(hasDistinctMetric ? [metricLabel] : [])]);
    for (const model of [...models].sort((a,b) => (item.scores[b.id] ?? -Infinity) - (item.scores[a.id] ?? -Infinity))) {
      const scoreRow = node('tr', null, model.id === 'jev' ? 'reference-row' : '');
      scoreRow.append(rowHeader(model.name), visionScoreCell(model.id, item.scores));
      if (hasDistinctMetric) scoreRow.append(visionScoreCell(model.id, item.nativeScores,
        item.nativeMetric === 'mme_score' ? value => `${value.toFixed(2)} / 2,000` : percent));
      scores.append(scoreRow);
    }
    body.append(scrollTable(table, `${item.project} vision scores`), node('h4', 'An actual image question'));
    const figure = node('figure', null, 'vision-example');
    const image = node('img', null, item.id === 'vision-cifar10' ? 'cifar-preview' : ''); image.src = item.example.image; image.alt = `Evaluation image for ${item.project}, case ${item.example.id}`; image.loading = 'lazy';
    const caption = node('figcaption', `Case ${item.example.id}. Source: ${item.project}. Original evaluated PNG, no additional preprocessing. ${item.id === 'vision-cifar10' ? 'Original resolution: 32 × 32 pixels; enlarged with crisp pixels, not generated detail. ' : ''}`);
    const sourceURL = item.source.url || (item.source.repository ? `https://huggingface.co/datasets/${item.source.repository}` : null);
    if (sourceURL?.startsWith('https://')) { const link = node('a', 'Dataset source and terms ↗'); link.href = sourceURL; caption.append(link); }
    figure.append(image, caption); body.append(figure, questionBlock(item.example.question, item.example.gold));
    body.append(node('p', `${item.example.selection} The answer shown is dataset gold, not a model output.`));
    const provenance = node('details'); provenance.append(node('summary', 'Source, image hash and answer options'),
      pre({ source: item.source, imageSha256: item.example.imageSha256, options: item.example.options }));
    body.append(provenance); details.append(body); $('vision-items').append(details);
  }

  let examples = null; let filtered = []; let loading = false;
  function renderExample() {
    const id = $('example-id').value; const index = filtered.findIndex(row => row.id === id); const example = filtered[index];
    $('previous-example').disabled = index <= 0;
    $('next-example').disabled = index < 0 || index === filtered.length - 1;
    $('public-example').replaceChildren();
    if (!example) return;
    const box = node('article', null, 'eval-example');
    box.append(node('h4', `${example.id} · ${example.family.replaceAll('_', ' ')}`), contextBlock(example.state),
      node('h4', 'Question / permitted answers'), pre(example.question), node('p', `Gold answer: ${pretty(example.expected)}`, 'gold-answer'));
    const table = node('table'); const body = tableHeader(table, ['Model', 'Recorded answer', 'Outcome']);
    for (const model of [...models].sort((a,b) => Number(b.id === 'jev') - Number(a.id === 'jev'))) {
      const answer = example.answers[model.id];
      const row = node('tr', null, model.id === 'jev' ? 'reference-row' : answer.correct && !example.answers.jev.correct ? 'winner-row' : '');
      const outcome = node('td', answer.correct ? 'Correct' : 'Incorrect');
      markComparison(outcome, Number(answer.correct), Number(example.answers.jev.correct), model.id === 'jev', false);
      row.append(rowHeader(model.id === 'jev' ? `${model.name} · baseline` : model.name), node('td', answer.predicted ?? 'Label not published'), outcome); body.append(row);
    }
    box.append(scrollTable(table, 'Recorded model answers'));
    $('public-example').append(box);
    $('examples-status').textContent = `${filtered.length} matching questions · showing ${index + 1} of ${filtered.length}.`;
  }
  function filterPublic() {
    if (!examples) return;
    const previous = $('example-id').value;
    filtered = filterExamples(examples, $('example-tier').value, $('example-type').value, $('example-outcome').value);
    const select = $('example-id'); select.replaceChildren(); select.disabled = !filtered.length;
    for (const row of filtered) { const option = node('option', row.id); option.value = row.id; select.append(option); }
    if (filtered.some(row => row.id === previous)) select.value = previous;
    $('examples-status').textContent = `${filtered.length} matching questions.`;
    renderExample();
  }
  async function loadExamples() {
    if (examples || loading) return;
    loading = true; $('examples-status').textContent = 'Loading the 231 source questions…';
    try {
      examples = await fetchJSON('assets/evaluations/public-examples.json');
      if (!Array.isArray(examples) || examples.length !== 231) throw new Error('Invalid examples');
      filterPublic();
    } catch {
      examples = null;
      showRetry($('examples-status'), 'Could not load examples.', loadExamples);
    } finally { loading = false; }
  }
  $('public-set').addEventListener('toggle', () => { if ($('public-set').open) loadExamples(); });
  if ($('public-set').open) loadExamples();
  for (const id of ['example-tier', 'example-type', 'example-outcome']) $(id).addEventListener('change', filterPublic);
  $('example-id').addEventListener('change', renderExample);
  for (const [id, step] of [['previous-example', -1], ['next-example', 1]]) {
    $(id).addEventListener('click', () => { $('example-id').selectedIndex += step; renderExample(); });
  }
  function openAnchor() {
    const id = location.hash.slice(1);
    if (['public-set', 'decision-set', 'vision-set', 'methodology'].includes(id)) $(id).open = true;
  }
  window.addEventListener('hashchange', openAnchor); openAnchor();
  // Reopen a section even if its hash is already selected and the reader closed it.
  document.querySelectorAll('.eval-jumps a').forEach(link => link.addEventListener('click', () => {
    $(link.hash.slice(1)).open = true;
  }));
}
if (typeof document !== 'undefined') init();
