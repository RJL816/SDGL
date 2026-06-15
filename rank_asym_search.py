import argparse
import csv
import os
import re


PAPER_AUDIO = 63.12
PAPER_VISUAL = 69.11
PAPER_FUSION = 77.48
SENTINEL = ['1000', '1000', '1000']


def load_summary(path):
    values = {}
    with open(path, newline='') as csvfile:
        for row in csv.reader(csvfile):
            if not row or row[0] == 'metric':
                continue
            values[row[0]] = row[1] if len(row) > 1 else ''
            if len(row) > 2 and row[2]:
                values[row[0] + '_epoch'] = row[2]
    return values


def find_accuracy_csv(result_dir):
    for name in sorted(os.listdir(result_dir)):
        if not name.endswith('.csv'):
            continue
        if name == 'run_summary.csv' or name.endswith('_selector.csv'):
            continue
        return os.path.join(result_dir, name)
    return None


def find_selector_csv(result_dir):
    for name in sorted(os.listdir(result_dir)):
        if name.endswith('_selector.csv'):
            return os.path.join(result_dir, name)
    return None


def load_effective_selector(result_dir):
    selector_path = find_selector_csv(result_dir)
    if selector_path is None:
        return None
    with open(selector_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        first_row = next(reader, None)
    if first_row is None:
        return None
    return {
        'tau_a': float(first_row['tau_audio']),
        'lambda_a': float(first_row['lambda_audio']),
        'tau_v': float(first_row['tau_visual']),
        'lambda_v': float(first_row['lambda_visual']),
    }


def load_accuracy_at_epoch(path, target_epoch):
    rows = []
    with open(path, newline='') as csvfile:
        for row in csv.reader(csvfile):
            if len(row) < 3 or row[:3] == SENTINEL:
                continue
            rows.append([float(row[0]), float(row[1]), float(row[2])])
    if target_epoch >= len(rows):
        raise IndexError('Epoch {} is missing from {}'.format(target_epoch, path))
    fusion, audio, visual = rows[target_epoch]
    return audio * 100.0, visual * 100.0, fusion * 100.0


def summary_float(summary, specific_key, fallback_key):
    value = summary.get(specific_key, '')
    if value in ('', 'N/A'):
        value = summary[fallback_key]
    return float(value)


def parse_params(name, summary, effective_selector=None):
    if effective_selector is not None:
        seed_match = re.search(r'_seed(?P<seed>\d+)', name)
        return {
            **effective_selector,
            'seed': float(seed_match.group('seed')) if seed_match else 0.0,
        }
    pattern = (
        r'taua(?P<tau_a>[\d.]+)_lambdaa(?P<lambda_a>[\d.]+)_'
        r'tauv(?P<tau_v>[\d.]+)_lambdav(?P<lambda_v>[\d.]+)_seed(?P<seed>\d+)'
    )
    match = re.search(pattern, name)
    if match:
        return {key: float(value) for key, value in match.groupdict().items()}
    return {
        'tau_a': summary_float(summary, 'selector_tau_audio', 'selector_tau'),
        'lambda_a': summary_float(summary, 'selector_lambda_audio', 'selector_lambda'),
        'tau_v': summary_float(summary, 'selector_tau_visual', 'selector_tau'),
        'lambda_v': summary_float(summary, 'selector_lambda_visual', 'selector_lambda'),
        'seed': 0.0,
    }


def collect_result_dir(result_dir):
    name = os.path.basename(os.path.normpath(result_dir))
    summary_path = os.path.join(result_dir, 'run_summary.csv')
    accuracy_path = find_accuracy_csv(result_dir)
    if not os.path.isfile(summary_path) or accuracy_path is None:
        return None

    summary = load_summary(summary_path)
    best_epoch = int(summary['best_fusion_acc_epoch'])
    audio, visual, fusion = load_accuracy_at_epoch(accuracy_path, best_epoch)
    params = parse_params(name, summary, load_effective_selector(result_dir))
    margins = {
        'audio_margin': audio - PAPER_AUDIO,
        'visual_margin': visual - PAPER_VISUAL,
        'fusion_margin': fusion - PAPER_FUSION,
    }
    return {
        'name': name,
        **params,
        'epoch': best_epoch,
        'audio': audio,
        'visual': visual,
        'fusion': fusion,
        **margins,
        'all_above_paper': all(value >= 0 for value in margins.values()),
        'minimum_margin': min(margins.values()),
    }


def collect_results(results_root, prefix, baseline_dirs):
    records = []
    for name in sorted(os.listdir(results_root)):
        if not name.startswith(prefix):
            continue
        result_dir = os.path.join(results_root, name)
        record = collect_result_dir(result_dir)
        if record is not None:
            records.append(record)
    for result_dir in baseline_dirs:
        record = collect_result_dir(result_dir)
        if record is not None and all(item['name'] != record['name'] for item in records):
            records.append(record)
    return records


def ranking_key(record):
    return (
        record['all_above_paper'],
        record['fusion'] if record['all_above_paper'] else record['minimum_margin'],
        record['minimum_margin'],
        record['fusion'],
    )


def write_outputs(records, results_root, output_stem):
    records = sorted(records, key=ranking_key, reverse=True)
    csv_path = os.path.join(results_root, output_stem + '.csv')
    txt_path = os.path.join(results_root, output_stem + '.txt')
    fields = [
        'rank', 'name', 'seed',
        'tau_a', 'lambda_a', 'tau_v', 'lambda_v',
        'epoch', 'audio', 'visual', 'fusion',
        'audio_margin', 'visual_margin', 'fusion_margin',
        'minimum_margin', 'all_above_paper',
    ]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            writer.writerow({'rank': index, **record})

    lines = [
        'Joint Asymmetric Search Ranking',
        'Selection rule: first require Audio/Visual/Fusion all above paper; '
        'then maximize Fusion. If none is feasible, maximize the worst margin.',
        'Paper: Audio=63.12, Visual=69.11, Fusion=77.48',
        '',
    ]
    for index, record in enumerate(records, start=1):
        lines.append(
            '#{rank} {name}: A={audio:.2f} ({audio_margin:+.2f}), '
            'V={visual:.2f} ({visual_margin:+.2f}), '
            'F={fusion:.2f} ({fusion_margin:+.2f}), epoch={epoch}, '
            'tau_a={tau_a:.2f}, lambda_a={lambda_a:.2f}, '
            'tau_v={tau_v:.2f}, lambda_v={lambda_v:.2f}, feasible={feasible}'.format(
                rank=index,
                feasible='yes' if record['all_above_paper'] else 'no',
                **record
            )
        )
    with open(txt_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return txt_path, csv_path, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='./results')
    parser.add_argument('--prefix', default='cramed_sdgl_alpha4_joint_')
    parser.add_argument('--baseline_dir', action='append', default=[])
    parser.add_argument('--output_stem', default='joint_asym_search_ranking')
    args = parser.parse_args()

    records = collect_results(args.results_root, args.prefix, args.baseline_dir)
    if not records:
        raise RuntimeError('No completed search results found for prefix {}'.format(args.prefix))
    txt_path, csv_path, lines = write_outputs(records, args.results_root, args.output_stem)
    print('\n'.join(lines))
    print('\nSaved:')
    print(txt_path)
    print(csv_path)


if __name__ == '__main__':
    main()
