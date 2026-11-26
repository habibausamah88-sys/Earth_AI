"""Controlled studies: one factor at a time, validation only; no test tuning."""
import argparse
import json
import numpy as np
from configs import config as cfg
from src.train import run_training

STUDIES = {
    'lr': ('lr', [0.0001, 0.001, 0.01]),
    'batch': ('batch_size', [32, 64, 128]),
    'optimizer': ('optimizer', ['sgd', 'sgd_momentum', 'adam']),
    'architecture': ('dense_units', [64, 128, 256]),
    'ablation_aug': ('augment', [True, False]),
    'ablation_dropout': ('dropout', [0.4, 0.0]),
    'ablation_weights': ('class_weights', [True, False]),
    'ablation_bn': ('batch_norm', [True, False]),
}


def run_study(name, author, data_dir, smoke=False, epochs=None):
    factor, values = STUDIES[name]
    ref = dict(lr=0.001, batch_size=64, optimizer='adam', augment=True,
               dense_units=128, dropout=0.4, class_weights=True, batch_norm=True,
               epochs=epochs, max_per_class=None, config_tag=name)
    if smoke:
        ref.update(epochs=1, max_per_class=60, config_tag='smoke')
    rows = []
    for value in values:
        opts = {**ref, factor: value}
        result = run_training('model_a', data_dir, author=author, notes=f'{name}: {factor}={value}', **opts)
        rows.append(dict(factor=factor, value=value, **result))
    cfg.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    (cfg.EXPERIMENTS_DIR / f'{name}{"_smoke" if smoke else ""}.json').write_text(json.dumps(rows, indent=2))
    return max(rows, key=lambda r: r['val_macro_f1'])


def final_seeds(author, data_dir, run_name=None):
    from src.data import scan_dataset, dataset_signature
    paths, _, _ = scan_dataset(data_dir)
    signature = dataset_signature(paths)
    entries = json.loads((cfg.MODELS_DIR / 'manifest.json').read_text())
    candidates = [e for e in entries if e.get('eligible') and e.get('dataset_signature') == signature and e['model_key'] != 'baseline_mlp' and e.get('config_tag') != 'final_seed']
    if run_name:
        candidates = [e for e in candidates if e['run_name'] == run_name]
    if not candidates:
        raise ValueError('Train full models before final seeds.')
    best = max(candidates, key=lambda e: e['val_macro_f1'])
    opts = {k: best[k] for k in ('lr','batch_size','optimizer','augment','epochs','dropout','dense_units','batch_norm','class_weights','pretrained')}
    results = []
    for seed in cfg.TUNING['final_seeds']:
        parent = None
        if best['model_key'] == 'model_c':
            # Retrain frozen stage with each seed so variation includes the complete pipeline.
            old = next(e for e in entries if e['run_name'] == best['parent_run'])
            b_opts = {k: old[k] for k in ('lr','batch_size','optimizer','augment','epochs','class_weights','pretrained')}
            b = run_training('model_b', data_dir, author=author, seed=seed, config_tag='seed_parent', **b_opts)
            parent = b['run_name']
        results.append(run_training(best['model_key'], data_dir, author=author, seed=seed, config_tag='final_seed', parent_run=parent, **opts))
    summary = {'selected_configuration': best, 'runs': results,
               'deployment_run': results[0]['run_name']}
    for metric in ('val_accuracy', 'val_macro_f1'):
        values = [r[metric] for r in results]
        summary[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
    (cfg.EXPERIMENTS_DIR / 'final_seeds.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=[*STUDIES, 'seeds', 'all'])
    ap.add_argument('--author', default='Experiment engineer')
    ap.add_argument('--data-dir', default=str(cfg.DATA_DIR))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--run-name')
    args = ap.parse_args()
    if args.study == 'seeds':
        if args.smoke: ap.error('Final seed statistics require full data.')
        final_seeds(args.author, args.data_dir, args.run_name)
    else:
        for study in (STUDIES if args.study == 'all' else [args.study]):
            print(run_study(study, args.author, args.data_dir, args.smoke, args.epochs))

if __name__ == '__main__':
    main()
