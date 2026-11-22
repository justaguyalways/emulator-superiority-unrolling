"""Audit the completed experiment grid without initializing an ML runtime."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'extension_2026_11_22'


def identity(c):
    return (c['tag'], 'burgers' if c['convection_difficulty'] else
            ('explicit' if c['linear_difficulties'][1] == -.9 else 'implicit'),
            'varied' if c['ic_train'].startswith('indamp') else 'unit',
            c['train_mode'], c['num_unrolled_steps'], c['seed_init'])


def main():
    expected = set()
    for solver, heights, n, seed in itertools.product(
            ['explicit', 'implicit'], ['unit', 'varied'], [1, 2, 5], range(3)):
        expected.add(('extension_conv', solver, heights, 'sup', n, seed))
    for solver, n, seed in itertools.product(['explicit', 'implicit'], [1, 2, 5], range(3)):
        expected.add(('extension_fixedH5', solver, 'unit', 'sup', n, seed))
    for (mode, n), seed in itertools.product(
            [('sup', 1), ('sup', 2), ('sup', 5), ('div_wig', 2)], range(3)):
        expected.add(('extension_burgers', 'burgers', 'unit', mode, n, seed))

    rows = []
    for folder in sorted((OUT / 'runs').iterdir()):
        path = folder / 'summary.json'
        if not path.exists():
            path = folder / 'failure.json'
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        if s['config']['tag'] in {'extension_conv', 'extension_fixedH5', 'extension_burgers'}:
            rows.append((folder, s, path.name == 'summary.json'))
    identities = [identity(s['config']) for _, s, _ in rows]
    assert len(identities) == len(set(identities)), 'Duplicate experimental cells'
    assert set(identities) == expected, {'missing': sorted(expected-set(identities)),
                                         'extra': sorted(set(identities)-expected)}
    sources = OUT / 'source_snapshot'
    if not sources.exists():
        sources = ROOT / 'src'
    source_hashes = {name: hashlib.sha256((sources/name).read_bytes()).hexdigest()
                     for name in ['extension_study.py', 'emusup.py', 'fast_models.py']}
    provenance = set()
    baselines = {}
    baseline_max_difference = {}
    nonfinite = []
    training_failures = []
    for folder, s, completed in rows:
        c = s['config']
        assert s['provenance']['source_sha256'] == source_hashes, folder.name
        provenance.add(json.dumps(s['provenance'], sort_keys=True))
        budget = 100000 if c['convection_difficulty'] else 10000
        assert c['num_training_steps'] == budget
        assert (c['seed_train'], c['seed_test'], c['seed_shuffle']) == (10, 20, 30)
        assert c['seed_init'] in [0, 1, 2]
        if not completed:
            training_failures.append({'name': folder.name, 'reason': s['reason']})
            continue
        assert s['completed_updates'] == budget, folder.name
        with (folder/'metrics.csv').open() as f:
            metrics = list(csv.DictReader(f))
        assert len(metrics) == c['temporal_horizon_test']
        assert [int(r['time_step']) for r in metrics] == list(range(1, len(metrics)+1))
        teacher = [float(r['error_solver']) for r in metrics]
        model = [float(r['error_emulator']) for r in metrics]
        assert all(math.isfinite(v) and v > 0 for v in teacher)
        key = json.dumps({k:c[k] for k in ['linear_difficulties', 'convection_difficulty',
            'num_points', 'train_simulator', 'test_simulator', 'ic_test', 'seed_test',
            'num_samples_test', 'temporal_horizon_test', 'metric_test']}, sort_keys=True)
        if key not in baselines:
            baselines[key] = teacher
        delta = max(abs(a-b) for a,b in zip(teacher, baselines[key]))
        assert delta <= 2e-6 * max(teacher), (folder.name, delta)
        baseline_max_difference[key] = max(delta, baseline_max_difference.get(key, 0))
        if not all(math.isfinite(v) for v in model):
            nonfinite.append(folder.name)
        else:
            h = 20 if c['convection_difficulty'] else 100
            ratio = sum(model[:h])/sum(teacher[:h])
            assert math.isclose(ratio, s[f'integrated_ratio_1_{h}'], rel_tol=2e-6)
        for name in ['model.eqx', 'checkpoint.eqx', 'checkpoint.json', 'loss.npy',
                     'per_sample_errors.npz']:
            assert (folder/name).exists(), (folder.name, name)
    assert len(provenance) == 1, 'Mixed primary numerical environments'
    manifest = json.loads((OUT/'historical_manifest.json').read_text())
    for r in manifest['runs']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256']
    report = {'passed': True, 'planned_primary_cells': len(expected),
        'observed_primary_cells': len(rows), 'completed_full_budget': len(rows)-len(training_failures),
        'training_failures_retained': training_failures, 'nonfinite_rollouts_retained': nonfinite,
        'common_primary_provenance_versions': len(provenance), 'executed_source_sha256': source_hashes,
        'physical_test_conditions': len(baselines),
        'maximum_teacher_baseline_error_difference': max(baseline_max_difference.values()),
        'unchanged_historical_summaries': len(manifest['runs']),
        'notes': ['Primary ratios recalculated from per-step error CSVs.',
                  'All initialization seeds share the same data split.',
                  'The separate runtime calibration is excluded from primary grid counts.']}
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
