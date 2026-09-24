"""Native JevBench requests and unweighted accuracy using pinned upstream scoring."""
import copy
import math
from types import SimpleNamespace
from vendor.jevbench.scoring import score_task


def request_for(row, model):
    q = row['question']
    question = {'type': q['type'], 'instructions': q['instructions']}
    if q.get('criteria') is not None:
        question['criteria'] = copy.deepcopy(q['criteria'])
    return {'model': model, 'state': copy.deepcopy(row['state']),
            'questions': {'decision': question}}


def validate(rows):
    if not rows:
        raise ValueError('JevBench dataset is empty')
    seen = set()
    for row in rows:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('JevBench rows require unique nonempty string IDs')
        seen.add(identifier)
        labels = row.get('labels')
        if (not isinstance(labels, list) or len(labels) < 2
                or any(not isinstance(x, str) or not x for x in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('Invalid JevBench labels')
        expected = row.get('expected')
        if expected is not None and (type(expected) not in (str, int) or str(expected) not in labels):
            raise ValueError('Expected answer must be a listed label or null')
        if not isinstance(row.get('state'), (str, dict, list)):
            raise ValueError('Invalid JevBench state')
        q = row.get('question', {})
        if not isinstance(q, dict) or not isinstance(q.get('instructions'), str) or not q['instructions']:
            raise ValueError('Missing JevBench question instructions')
        kind, criteria = q.get('type'), q.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or set(criteria) != set(labels):
                raise ValueError('Choice criteria must match labels')
        elif kind == 'score':
            if not isinstance(criteria, list) or labels != [str(i) for i in range(len(criteria))]:
                raise ValueError('Score labels must be ordered zero-based levels')
        elif kind == 'noul':
            if set(labels) != {'yes', 'no'}:
                raise ValueError('Noul labels must be yes/no')
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}):
                raise ValueError('Noul criteria must use true/false')
        else:
            raise ValueError('Unsupported JevBench primitive')
        for key in ('tier', 'family'):
            if not isinstance(row.get(key, 'unspecified'), str):
                raise ValueError(f'Invalid {key}')


def parse_response(row, response):
    answer = response['answers']['decision']
    kind = row['question']['type']
    if answer.get('type') != kind:
        raise ValueError('Response primitive differs from requested primitive')
    if kind == 'noul':
        p = answer['noul']
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid Noul probability')
        probs = {'yes': p, 'no': 1 - p}
    else:
        probs = answer['probabilities']
    scored = score_task(probs, SimpleNamespace(**row))
    if not scored['valid']:
        raise ValueError(scored['error'])
    return {'probabilities': scored['probs'], 'predicted': scored['predicted'],
            'correct': scored['correct'], 'strict_valid': scored['strict_valid'],
            'renormalized': scored['renormalized']}


def summarize(rows, records):
    if len(rows) != len(records) or any(r['id'] != p['id'] for r, p in zip(rows, records)):
        raise ValueError('JevBench records must match dataset order and length')

    def metrics(pairs):
        scored = [(r, p) for r, p in pairs if r.get('expected') is not None]
        correct = sum(p.get('correct') is True for _, p in scored)
        return {'rows': len(pairs), 'scorable_rows': len(scored), 'correct': correct,
                'accuracy': correct / len(scored) if scored else None,
                'successful_rows': sum('probabilities' in p for _, p in pairs),
                'failed_rows': sum('probabilities' not in p for _, p in pairs)}

    pairs = list(zip(rows, records))
    result = metrics(pairs)
    result['unscorable_rows'] = len(rows) - result['scorable_rows']
    for name, key in (('by_tier', 'tier'), ('by_family', 'family'), ('by_primitive', None)):
        groups = {}
        for row, record in pairs:
            value = row.get(key, 'unspecified') if key else row['question']['type']
            groups.setdefault(value, []).append((row, record))
        result[name] = {label: metrics(group) for label, group in sorted(groups.items())}
    return result
