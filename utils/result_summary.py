import csv
import os


SENTINEL_ROW = ['1000', '1000', '1000']


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_percent(value):
    if value is None:
        return 'N/A'
    return '{:.2f}%'.format(value * 100.0)


def _format_float(value):
    if value is None:
        return 'N/A'
    return '{:.6f}'.format(value)


def load_accuracy_rows(log_path):
    rows = []
    if not os.path.exists(log_path):
        return rows

    with open(log_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) < 3:
                continue
            if row[:3] == SENTINEL_ROW:
                continue

            fusion = _safe_float(row[0])
            audio = _safe_float(row[1])
            visual = _safe_float(row[2])
            if fusion is None or audio is None or visual is None:
                continue

            rows.append({
                'epoch': len(rows),
                'fusion': fusion,
                'audio': audio,
                'visual': visual,
            })

    return rows


def load_selector_epoch_rows(selector_path):
    rows = []
    if not selector_path or not os.path.exists(selector_path):
        return rows

    with open(selector_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('phase') != 'epoch':
                continue

            parsed = {'epoch': int(row['epoch'])}
            for key, value in row.items():
                if key in ['phase', 'epoch', 'step']:
                    continue
                parsed[key] = _safe_float(value)
            rows.append(parsed)

    return rows


def summarize_accuracy_rows(rows):
    if not rows:
        return None

    best_fusion = max(rows, key=lambda row: row['fusion'])
    best_audio = max(rows, key=lambda row: row['audio'])
    best_visual = max(rows, key=lambda row: row['visual'])
    last_row = rows[-1]

    return {
        'num_epochs': len(rows),
        'best_fusion': best_fusion,
        'best_audio': best_audio,
        'best_visual': best_visual,
        'last': last_row,
    }


def summarize_selector_rows(rows):
    if not rows:
        return None

    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    last_row = rows[-1]
    return {
        'num_epochs': len(rows),
        'mean_sim_audio': mean('sim_audio'),
        'mean_sim_visual': mean('sim_visual'),
        'mean_sample_useful_ratio_audio': mean('sample_useful_ratio_audio'),
        'mean_sample_useful_ratio_visual': mean('sample_useful_ratio_visual'),
        'mean_beta_audio': mean('beta_audio'),
        'mean_beta_visual': mean('beta_visual'),
        'last_sim_audio': last_row.get('sim_audio'),
        'last_sim_visual': last_row.get('sim_visual'),
        'last_sample_useful_ratio_audio': last_row.get('sample_useful_ratio_audio'),
        'last_sample_useful_ratio_visual': last_row.get('sample_useful_ratio_visual'),
        'last_beta_audio': last_row.get('beta_audio'),
        'last_beta_visual': last_row.get('beta_visual'),
    }


def build_summary_lines(result_dir, meta, accuracy_summary, selector_summary=None):
    lines = []
    lines.append('Result Summary')
    lines.append('Result Dir: {}'.format(result_dir))
    if meta:
        lines.append(
            'Config: dataset={dataset}, modality={modality}, grad_strategy={grad_strategy}, '
            'alpha={alpha}, tau={selector_tau}, lambda={selector_lambda}, start_epoch={selector_start_epoch}'.format(
                dataset=meta.get('dataset', 'N/A'),
                modality=meta.get('modality', 'N/A'),
                grad_strategy=meta.get('grad_strategy', 'N/A'),
                alpha=meta.get('alpha', 'N/A'),
                selector_tau=meta.get('selector_tau', 'N/A'),
                selector_lambda=meta.get('selector_lambda', 'N/A'),
                selector_start_epoch=meta.get('selector_start_epoch', 'N/A'),
            )
        )
        lines.append(
            'Effective Selector: tau_audio={tau_audio}, tau_visual={tau_visual}, '
            'lambda_audio={lambda_audio}, lambda_visual={lambda_visual}'.format(
                tau_audio=meta.get('selector_tau_audio', meta.get('selector_tau', 'N/A')),
                tau_visual=meta.get('selector_tau_visual', meta.get('selector_tau', 'N/A')),
                lambda_audio=meta.get('selector_lambda_audio', meta.get('selector_lambda', 'N/A')),
                lambda_visual=meta.get('selector_lambda_visual', meta.get('selector_lambda', 'N/A')),
            )
        )

    if accuracy_summary is None:
        lines.append('No accuracy rows were found.')
        return lines

    best_fusion = accuracy_summary['best_fusion']
    best_audio = accuracy_summary['best_audio']
    best_visual = accuracy_summary['best_visual']
    last_row = accuracy_summary['last']

    lines.append('Epochs Logged: {}'.format(accuracy_summary['num_epochs']))
    lines.append(
        'Best Fusion Acc: {} at epoch {} (audio={}, visual={})'.format(
            _format_percent(best_fusion['fusion']),
            best_fusion['epoch'],
            _format_percent(best_fusion['audio']),
            _format_percent(best_fusion['visual']),
        )
    )
    lines.append(
        'Best Audio Acc: {} at epoch {}'.format(
            _format_percent(best_audio['audio']),
            best_audio['epoch'],
        )
    )
    lines.append(
        'Best Visual Acc: {} at epoch {}'.format(
            _format_percent(best_visual['visual']),
            best_visual['epoch'],
        )
    )
    lines.append(
        'Last Epoch Acc: fusion={}, audio={}, visual={}'.format(
            _format_percent(last_row['fusion']),
            _format_percent(last_row['audio']),
            _format_percent(last_row['visual']),
        )
    )

    if selector_summary is not None:
        lines.append(
            'Mean Selector Stats: sim_audio={}, sim_visual={}, useful_audio={}, useful_visual={}, beta_audio={}, beta_visual={}'.format(
                _format_float(selector_summary['mean_sim_audio']),
                _format_float(selector_summary['mean_sim_visual']),
                _format_percent(selector_summary['mean_sample_useful_ratio_audio']),
                _format_percent(selector_summary['mean_sample_useful_ratio_visual']),
                _format_float(selector_summary['mean_beta_audio']),
                _format_float(selector_summary['mean_beta_visual']),
            )
        )
        lines.append(
            'Last Selector Stats: sim_audio={}, sim_visual={}, useful_audio={}, useful_visual={}, beta_audio={}, beta_visual={}'.format(
                _format_float(selector_summary['last_sim_audio']),
                _format_float(selector_summary['last_sim_visual']),
                _format_percent(selector_summary['last_sample_useful_ratio_audio']),
                _format_percent(selector_summary['last_sample_useful_ratio_visual']),
                _format_float(selector_summary['last_beta_audio']),
                _format_float(selector_summary['last_beta_visual']),
            )
        )

    return lines


def build_machine_summary_rows(meta, accuracy_summary, selector_summary=None):
    rows = [['metric', 'value', 'epoch']]
    if meta:
        for key in [
            'dataset', 'modality', 'grad_strategy', 'alpha',
            'selector_tau', 'selector_lambda',
            'selector_tau_audio', 'selector_tau_visual',
            'selector_lambda_audio', 'selector_lambda_visual',
            'selector_start_epoch',
        ]:
            rows.append([key, meta.get(key, 'N/A'), ''])

    if accuracy_summary is None:
        rows.append(['status', 'no_accuracy_rows', ''])
        return rows

    rows.append(['best_fusion_acc', accuracy_summary['best_fusion']['fusion'], accuracy_summary['best_fusion']['epoch']])
    rows.append(['best_audio_acc', accuracy_summary['best_audio']['audio'], accuracy_summary['best_audio']['epoch']])
    rows.append(['best_visual_acc', accuracy_summary['best_visual']['visual'], accuracy_summary['best_visual']['epoch']])
    rows.append(['last_fusion_acc', accuracy_summary['last']['fusion'], accuracy_summary['last']['epoch']])
    rows.append(['last_audio_acc', accuracy_summary['last']['audio'], accuracy_summary['last']['epoch']])
    rows.append(['last_visual_acc', accuracy_summary['last']['visual'], accuracy_summary['last']['epoch']])

    if selector_summary is not None:
        rows.append(['mean_sim_audio', selector_summary['mean_sim_audio'], ''])
        rows.append(['mean_sim_visual', selector_summary['mean_sim_visual'], ''])
        rows.append(['mean_sample_useful_ratio_audio', selector_summary['mean_sample_useful_ratio_audio'], ''])
        rows.append(['mean_sample_useful_ratio_visual', selector_summary['mean_sample_useful_ratio_visual'], ''])
        rows.append(['mean_beta_audio', selector_summary['mean_beta_audio'], ''])
        rows.append(['mean_beta_visual', selector_summary['mean_beta_visual'], ''])
        rows.append(['last_beta_audio', selector_summary['last_beta_audio'], ''])
        rows.append(['last_beta_visual', selector_summary['last_beta_visual'], ''])

    return rows


def write_run_summary(result_dir, log_path, selector_path=None, meta=None):
    accuracy_rows = load_accuracy_rows(log_path)
    selector_rows = load_selector_epoch_rows(selector_path)

    accuracy_summary = summarize_accuracy_rows(accuracy_rows)
    selector_summary = summarize_selector_rows(selector_rows)

    summary_lines = build_summary_lines(result_dir, meta or {}, accuracy_summary, selector_summary)
    text_path = os.path.join(result_dir, 'run_summary.txt')
    csv_path = os.path.join(result_dir, 'run_summary.csv')

    with open(text_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary_lines) + '\n')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(build_machine_summary_rows(meta or {}, accuracy_summary, selector_summary))

    return '\n'.join(summary_lines), text_path, csv_path
