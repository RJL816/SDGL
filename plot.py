import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


PAPER_RESULT = {
    'audio': 0.6312,
    'visual': 0.6911,
    'fusion': 0.7748,
}

SENTINEL_ROW = ['1000', '1000', '1000']


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def configure_plot_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#303030',
        'axes.labelcolor': '#303030',
        'axes.titlesize': 12,
        'axes.titleweight': 'semibold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'xtick.color': '#303030',
        'ytick.color': '#303030',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'savefig.bbox': 'tight',
    })


def save_figure(fig, output_dir, stem, dpi=300):
    for extension in ('png', 'pdf', 'svg'):
        save_kwargs = {'bbox_inches': 'tight'}
        if extension == 'png':
            save_kwargs['dpi'] = dpi
        fig.savefig(os.path.join(output_dir, '{}.{}'.format(stem, extension)), **save_kwargs)
    plt.close(fig)


def load_run_summary_csv(result_dir):
    path = os.path.join(result_dir, 'run_summary.csv')
    if not os.path.exists(path):
        return None

    metrics = {}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or row[0] == 'metric':
                continue
            metric = row[0]
            value = row[1] if len(row) > 1 else ''
            epoch = row[2] if len(row) > 2 else ''
            metrics[metric] = {'value': value, 'epoch': epoch}
    return metrics


def load_accuracy_curve(result_dir):
    curve_path = None
    for file_name in os.listdir(result_dir):
        if file_name.endswith('.csv') and '_selector' not in file_name and file_name != 'run_summary.csv':
            curve_path = os.path.join(result_dir, file_name)
            break

    if curve_path is None:
        return []

    rows = []
    with open(curve_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 3 or row[:3] == SENTINEL_ROW:
                continue
            rows.append({
                'epoch': len(rows),
                'fusion': float(row[0]),
                'audio': float(row[1]),
                'visual': float(row[2]),
            })
    return rows


def load_selector_epoch_rows(result_dir):
    selector_path = None
    for file_name in os.listdir(result_dir):
        if file_name.endswith('_selector.csv'):
            selector_path = os.path.join(result_dir, file_name)
            break

    if selector_path is None:
        return []

    rows = []
    with open(selector_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('phase') != 'epoch':
                continue
            rows.append({
                'epoch': int(row['epoch']),
                'loss_f': float(row['loss_f']),
                'loss_a': float(row['loss_a']),
                'loss_v': float(row['loss_v']),
                'sim_audio': float(row['sim_audio']),
                'sim_visual': float(row['sim_visual']),
                'sample_useful_ratio_audio': float(row['sample_useful_ratio_audio']),
                'sample_useful_ratio_visual': float(row['sample_useful_ratio_visual']),
                'beta_audio': float(row['beta_audio']),
                'beta_visual': float(row['beta_visual']),
            })
    return rows


def discover_result_dirs(results_root):
    result_dirs = []
    for name in sorted(os.listdir(results_root)):
        abs_dir = os.path.join(results_root, name)
        if not os.path.isdir(abs_dir):
            continue
        if os.path.exists(os.path.join(abs_dir, 'run_summary.csv')):
            result_dirs.append(abs_dir)
    return result_dirs


def build_run_records(results_root):
    records = []
    for result_dir in discover_result_dirs(results_root):
        summary = load_run_summary_csv(result_dir)
        curve = load_accuracy_curve(result_dir)
        selector_rows = load_selector_epoch_rows(result_dir)
        if summary is None or not curve:
            continue

        best_epoch = int(summary['best_fusion_acc']['epoch'])
        best_row = curve[best_epoch]
        tau_audio = summary.get('selector_tau_audio', summary.get('selector_tau', {})).get('value', 'N/A')
        tau_visual = summary.get('selector_tau_visual', summary.get('selector_tau', {})).get('value', 'N/A')
        lambda_audio = summary.get('selector_lambda_audio', summary.get('selector_lambda', {})).get('value', 'N/A')
        lambda_visual = summary.get('selector_lambda_visual', summary.get('selector_lambda', {})).get('value', 'N/A')
        tau_label = tau_audio if tau_audio == tau_visual else '{}/{}'.format(tau_audio, tau_visual)
        lambda_label = lambda_audio if lambda_audio == lambda_visual else '{}/{}'.format(lambda_audio, lambda_visual)

        record = {
            'name': os.path.basename(result_dir),
            'result_dir': result_dir,
            'summary': summary,
            'curve': curve,
            'selector_rows': selector_rows,
            'best_epoch': best_epoch,
            'best_fusion': float(summary['best_fusion_acc']['value']),
            'best_audio': float(summary['best_audio_acc']['value']),
            'best_visual': float(summary['best_visual_acc']['value']),
            'best_checkpoint_audio': best_row['audio'],
            'best_checkpoint_visual': best_row['visual'],
            'last_fusion': float(summary['last_fusion_acc']['value']),
            'tau': tau_label,
            'lambda': lambda_label,
        }
        records.append(record)
    records.sort(key=lambda item: item['best_fusion'], reverse=True)
    return records


def aggregate_selector_runs(result_dirs):
    all_rows = []
    for result_dir in result_dirs:
        rows = load_selector_epoch_rows(result_dir)
        if not rows:
            continue
        all_rows.append(rows)

    if not all_rows:
        return None

    min_len = min(len(rows) for rows in all_rows)
    keys = [
        'loss_f', 'loss_a', 'loss_v',
        'sim_audio', 'sim_visual',
        'sample_useful_ratio_audio', 'sample_useful_ratio_visual',
        'beta_audio', 'beta_visual',
    ]
    aggregated = {'epoch': list(range(min_len))}
    for key in keys:
        stacked = np.array([[rows[idx][key] for idx in range(min_len)] for rows in all_rows], dtype=float)
        aggregated[key + '_mean'] = stacked.mean(axis=0)
        ddof = 1 if stacked.shape[0] > 1 else 0
        aggregated[key + '_std'] = stacked.std(axis=0, ddof=ddof)

    tau_audio = []
    tau_visual = []
    lambda_audio = []
    lambda_visual = []
    for result_dir in result_dirs:
        summary = load_run_summary_csv(result_dir)
        if not summary:
            continue
        tau_audio.append(float(summary.get('selector_tau_audio', summary['selector_tau'])['value']))
        tau_visual.append(float(summary.get('selector_tau_visual', summary['selector_tau'])['value']))
        lambda_audio.append(float(summary.get('selector_lambda_audio', summary['selector_lambda'])['value']))
        lambda_visual.append(float(summary.get('selector_lambda_visual', summary['selector_lambda'])['value']))
    aggregated['tau_audio'] = float(np.mean(tau_audio))
    aggregated['tau_visual'] = float(np.mean(tau_visual))
    aggregated['lambda_audio'] = float(np.mean(lambda_audio))
    aggregated['lambda_visual'] = float(np.mean(lambda_visual))
    return aggregated


def plot_training_loss_multiseed(result_dirs, output_dir):
    aggregated = aggregate_selector_runs(result_dirs)
    if aggregated is None:
        return

    epochs = np.asarray(aggregated['epoch'])
    audio_mean = aggregated['loss_a_mean']
    audio_std = aggregated['loss_a_std']
    visual_mean = aggregated['loss_v_mean']
    visual_std = aggregated['loss_v_std']

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(epochs, audio_mean, color='#24476B', linewidth=2.2, label='Audio loss')
    ax.fill_between(
        epochs,
        np.maximum(audio_mean - audio_std, 1e-6),
        audio_mean + audio_std,
        color='#24476B',
        alpha=0.15,
        linewidth=0,
    )
    ax.semilogy(epochs, visual_mean, color='#E76F51', linewidth=2.2, label='Visual loss')
    ax.fill_between(
        epochs,
        np.maximum(visual_mean - visual_std, 1e-6),
        visual_mean + visual_std,
        color='#E76F51',
        alpha=0.15,
        linewidth=0,
    )
    ax.axvspan(18, 100, color='#24476B', alpha=0.04)
    ax.annotate(
        'Audio branch reaches near-zero loss',
        xy=(20, audio_mean[20]),
        xytext=(30, 0.035),
        arrowprops={'arrowstyle': '->', 'color': '#24476B', 'lw': 1.0},
        color='#24476B',
        fontsize=9,
    )
    ax.text(
        98,
        visual_mean[-1] * 1.18,
        'Visual: {:.3f}'.format(visual_mean[-1]),
        color='#E76F51',
        ha='right',
        fontsize=9,
    )
    ax.text(
        98,
        audio_mean[-1] * 0.78,
        'Audio: {:.4f}'.format(audio_mean[-1]),
        color='#24476B',
        ha='right',
        fontsize=9,
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss (log scale)')
    ax.set_title('Different Convergence Speeds Across Modalities')
    ax.set_xlim(0, 99)
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'training_loss_multiseed')


def plot_selector_similarity_threshold_multiseed(result_dirs, output_dir):
    aggregated = aggregate_selector_runs(result_dirs)
    if aggregated is None:
        return

    epochs = np.asarray(aggregated['epoch'])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for prefix, label, color in (
        ('sim_audio', 'Audio similarity', '#24476B'),
        ('sim_visual', 'Visual similarity', '#E76F51'),
    ):
        mean = aggregated[prefix + '_mean']
        std = aggregated[prefix + '_std']
        ax.plot(epochs, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    tau = aggregated['tau_audio']
    ax.axhline(tau, color='#555555', linestyle='--', linewidth=1.4, label=r'Shared threshold $\tau=0.15$')
    ax.fill_between(epochs, 0, tau, color='#777777', alpha=0.05)
    ax.text(98, tau - 0.045, 'Fusion gradient rejected', ha='right', color='#666666', fontsize=8.5)
    ax.annotate(
        'Audio falls below threshold',
        xy=(19, aggregated['sim_audio_mean'][19]),
        xytext=(30, 0.27),
        arrowprops={'arrowstyle': '->', 'color': '#24476B', 'lw': 1.0},
        color='#24476B',
        fontsize=9,
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature-gradient cosine similarity')
    ax.set_title('A Shared Threshold Produces Modality-Specific Decisions')
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 1.03)
    ax.legend(loc='upper right')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'selector_similarity_threshold_multiseed')


def plot_selector_activation_multiseed(result_dirs, output_dir):
    aggregated = aggregate_selector_runs(result_dirs)
    if aggregated is None:
        return

    epochs = np.asarray(aggregated['epoch'])
    lambda_audio = aggregated['lambda_audio']
    lambda_visual = aggregated['lambda_visual']
    audio_gate = aggregated['beta_audio_mean'] / lambda_audio
    visual_gate = aggregated['beta_visual_mean'] / lambda_visual
    audio_gate_std = aggregated['beta_audio_std'] / lambda_audio
    visual_gate_std = aggregated['beta_visual_std'] / lambda_visual

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            audio_gate,
            audio_gate_std,
            visual_gate,
            visual_gate_std,
            'Batch-level gate activation',
            'Fraction of batches with mean similarity > tau',
        ),
        (
            axes[1],
            aggregated['sample_useful_ratio_audio_mean'],
            aggregated['sample_useful_ratio_audio_std'],
            aggregated['sample_useful_ratio_visual_mean'],
            aggregated['sample_useful_ratio_visual_std'],
            'Sample-level useful ratio',
            'Fraction of samples with similarity > tau',
        ),
    )
    for ax, audio_mean, audio_std, visual_mean, visual_std, title, subtitle in panels:
        ax.plot(epochs, audio_mean, color='#24476B', linewidth=2.1, label='Audio')
        ax.fill_between(epochs, audio_mean - audio_std, audio_mean + audio_std,
                        color='#24476B', alpha=0.15, linewidth=0)
        ax.plot(epochs, visual_mean, color='#E76F51', linewidth=2.1, label='Visual')
        ax.fill_between(epochs, visual_mean - visual_std, visual_mean + visual_std,
                        color='#E76F51', alpha=0.15, linewidth=0)
        ax.set_title(title, pad=18)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=8, color='#666666')
        ax.set_xlabel('Epoch')
        ax.set_xlim(0, 99)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel('Ratio')
    axes[0].legend(loc='upper right')
    fig.suptitle('Shared Parameters Yield Unequal Gate Activation',
                 y=1.04, fontsize=12, fontweight='semibold')
    fig.tight_layout()
    save_figure(fig, output_dir, 'selector_activation_multiseed')


def plot_asymmetry_evidence_multiseed(result_dirs, output_dir):
    aggregated = aggregate_selector_runs(result_dirs)
    if aggregated is None:
        return

    epochs = np.asarray(aggregated['epoch'])
    colors = {'audio': '#24476B', 'visual': '#E76F51'}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

    ax = axes[0]
    for key, label, color in (
        ('loss_a', 'Audio', colors['audio']),
        ('loss_v', 'Visual', colors['visual']),
    ):
        mean = aggregated[key + '_mean']
        std = aggregated[key + '_std']
        ax.semilogy(epochs, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(epochs, np.maximum(mean - std, 1e-6), mean + std,
                        color=color, alpha=0.14, linewidth=0)
    ax.set_title('a  Optimization state', loc='left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss (log scale)')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.55, alpha=0.7)

    ax = axes[1]
    for key, label, color in (
        ('sim_audio', 'Audio', colors['audio']),
        ('sim_visual', 'Visual', colors['visual']),
    ):
        mean = aggregated[key + '_mean']
        std = aggregated[key + '_std']
        ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.axhline(aggregated['tau_audio'], color='#555555', linestyle='--', linewidth=1.2,
               label=r'$\tau=0.15$')
    ax.set_title('b  Gradient agreement', loc='left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine similarity')
    ax.set_ylim(0, 1.03)
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.55, alpha=0.7)

    ax = axes[2]
    audio_gate = aggregated['beta_audio_mean'] / aggregated['lambda_audio']
    visual_gate = aggregated['beta_visual_mean'] / aggregated['lambda_visual']
    ax.plot(epochs, audio_gate, color=colors['audio'], linewidth=2.0, label='Audio')
    ax.plot(epochs, visual_gate, color=colors['visual'], linewidth=2.0, label='Visual')
    ax.fill_between(
        epochs,
        audio_gate - aggregated['beta_audio_std'] / aggregated['lambda_audio'],
        audio_gate + aggregated['beta_audio_std'] / aggregated['lambda_audio'],
        color=colors['audio'],
        alpha=0.14,
        linewidth=0,
    )
    ax.fill_between(
        epochs,
        visual_gate - aggregated['beta_visual_std'] / aggregated['lambda_visual'],
        visual_gate + aggregated['beta_visual_std'] / aggregated['lambda_visual'],
        color=colors['visual'],
        alpha=0.14,
        linewidth=0,
    )
    ax.set_title('c  Gate consequence', loc='left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Batch gate activation rate')
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.55, alpha=0.7)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('Evidence for Modality-Asymmetric Gradient Gating', y=1.04,
                 fontsize=13, fontweight='semibold')
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_dir, 'asymmetry_evidence_multiseed')


def plot_multiseed_performance_vs_paper(result_dirs, output_dir):
    values = []
    for result_dir in result_dirs:
        summary = load_run_summary_csv(result_dir)
        curve = load_accuracy_curve(result_dir)
        if not summary or not curve:
            continue
        best_epoch = int(summary['best_fusion_acc']['epoch'])
        row = curve[best_epoch]
        values.append([row['audio'] * 100.0, row['visual'] * 100.0, row['fusion'] * 100.0])
    if not values:
        return

    values = np.asarray(values)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(3)
    paper = np.asarray([
        PAPER_RESULT['audio'] * 100.0,
        PAPER_RESULT['visual'] * 100.0,
        PAPER_RESULT['fusion'] * 100.0,
    ])
    labels = ['Audio', 'Visual', 'Fusion']
    x = np.arange(3)
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    paper_bars = ax.bar(x - width / 2, paper, width, color='#A9A2B8', label='Paper')
    current_bars = ax.bar(
        x + width / 2,
        means,
        width,
        yerr=stds,
        capsize=3,
        color='#2A9D8F',
        label='Symmetric SDGL (3 seeds)',
    )
    for bar, value in zip(paper_bars, paper):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.75,
            '{:.2f}'.format(value),
            ha='center',
            va='top',
            fontsize=8.5,
            color='white',
            fontweight='semibold',
        )
    for bar, value, std in zip(current_bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + 0.25,
            '{:.2f}'.format(value),
            ha='center',
            va='bottom',
            fontsize=8.5,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel('Accuracy at best fusion checkpoint (%)')
    ax.set_ylim(58, 81)
    ax.set_title('Three-Seed Performance Compared with the Paper')
    ax.legend(loc='upper left')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'multiseed_performance_vs_paper')


def plot_ablation_best_fusion(records, output_dir):
    names = [record['name'] for record in records]
    values = [record['best_fusion'] * 100.0 for record in records]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(records)), values, color='#3a6ea5')
    plt.axhline(PAPER_RESULT['fusion'] * 100.0, color='#d1495b', linestyle='--', label='Paper Fusion')
    plt.xticks(range(len(records)), names, rotation=30, ha='right')
    plt.ylabel('Best Fusion Accuracy (%)')
    plt.title('CREMAD SDGL Ablation: Best Fusion Accuracy')
    plt.legend()
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.05, '{:.2f}'.format(value),
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_best_fusion.png'), dpi=200)
    plt.close()


def plot_best_checkpoint_compare(best_record, output_dir):
    metrics = ['Audio', 'Visual', 'Fusion']
    paper_values = [
        PAPER_RESULT['audio'] * 100.0,
        PAPER_RESULT['visual'] * 100.0,
        PAPER_RESULT['fusion'] * 100.0,
    ]
    current_values = [
        best_record['best_checkpoint_audio'] * 100.0,
        best_record['best_checkpoint_visual'] * 100.0,
        best_record['best_fusion'] * 100.0,
    ]

    x = [0, 1, 2]
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.6))
    paper_bars = ax.bar(
        [v - width / 2 for v in x], paper_values, width=width,
        label='Paper DGL', color='#A9A2B8',
    )
    current_bars = ax.bar(
        [v + width / 2 for v in x], current_values, width=width,
        label='Symmetric SDGL, balanced seed', color='#2A9D8F',
    )
    for bars in (paper_bars, current_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.18,
                '{:.2f}'.format(value),
                ha='center',
                va='bottom',
                fontsize=8.5,
            )
    ax.set_xticks(x, metrics)
    ax.set_ylabel('Accuracy at best fusion checkpoint (%)')
    ax.set_ylim(60, 80)
    ax.set_title('Balanced Symmetric SDGL Run Compared with the Paper')
    ax.legend(loc='upper left')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'best_checkpoint_vs_paper')


def plot_best_run_accuracy_curves(best_record, output_dir):
    rows = best_record['curve']
    epochs = [row['epoch'] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, [row['audio'] * 100.0 for row in rows],
            color='#24476B', linewidth=2.0, label='Audio')
    ax.plot(epochs, [row['visual'] * 100.0 for row in rows],
            color='#E76F51', linewidth=2.0, label='Visual')
    ax.plot(epochs, [row['fusion'] * 100.0 for row in rows],
            color='#2A9D8F', linewidth=2.3, label='Fusion')
    ax.axvline(best_record['best_epoch'], color='#555555', linestyle='--', linewidth=1.2)
    ax.scatter(
        [best_record['best_epoch']] * 3,
        [
            best_record['best_checkpoint_audio'] * 100.0,
            best_record['best_checkpoint_visual'] * 100.0,
            best_record['best_fusion'] * 100.0,
        ],
        color=['#24476B', '#E76F51', '#2A9D8F'],
        s=34,
        zorder=4,
    )
    ax.text(
        best_record['best_epoch'] - 1,
        42,
        'Best fusion checkpoint\nepoch {}'.format(best_record['best_epoch']),
        ha='right',
        va='bottom',
        fontsize=8.5,
        color='#555555',
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Dynamics of the Balanced Symmetric Run')
    ax.set_xlim(0, len(rows) - 1)
    ax.set_ylim(10, 82)
    ax.legend(loc='lower right')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'best_run_accuracy_curves')


def plot_best_run_training_loss(best_record, output_dir):
    rows = best_record['selector_rows']
    if not rows:
        return

    epochs = [row['epoch'] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(epochs, [row['loss_a'] for row in rows],
                color='#24476B', linewidth=2.1, label='Audio loss')
    ax.semilogy(epochs, [row['loss_v'] for row in rows],
                color='#E76F51', linewidth=2.1, label='Visual loss')
    ax.semilogy(epochs, [row['loss_f'] for row in rows],
                color='#2A9D8F', linewidth=2.1, label='Fusion loss')
    ax.axvline(best_record['best_epoch'], color='#555555', linestyle='--', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy loss (log scale)')
    ax.set_title('Training Losses of the Balanced Symmetric Run')
    ax.set_xlim(0, len(rows) - 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'best_run_training_loss')


def plot_fusion_curves(records, output_dir, top_k=4):
    selected = records[:top_k]
    plt.figure(figsize=(9, 5))
    for record in selected:
        epochs = [row['epoch'] for row in record['curve']]
        fusion = [row['fusion'] * 100.0 for row in record['curve']]
        plt.plot(epochs, fusion, label=record['name'])
    plt.axhline(PAPER_RESULT['fusion'] * 100.0, color='#d1495b', linestyle='--', label='Paper Fusion')
    plt.xlabel('Epoch')
    plt.ylabel('Fusion Accuracy (%)')
    plt.title('Fusion Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_accuracy_curves.png'), dpi=200)
    plt.close()


def plot_selector_dynamics(best_record, output_dir):
    rows = best_record['selector_rows']
    if not rows:
        return

    epochs = [row['epoch'] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, [row['sim_audio'] for row in rows],
            label='Audio similarity', color='#24476B', linewidth=2.2)
    ax.plot(epochs, [row['sim_visual'] for row in rows],
            label='Visual similarity', color='#E76F51', linewidth=2.2)
    ax.axhline(0.15, color='#555555', linestyle='--', linewidth=1.2,
               label=r'Threshold $\tau=0.15$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature-gradient cosine similarity')
    ax.set_title('Gradient Similarity of the Balanced Symmetric Run')
    ax.set_xlim(0, 99)
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc='upper right')
    ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, 'selector_similarity_best_run')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    axes[0].plot(epochs, [row['beta_audio'] / 0.75 for row in rows],
                 label='Audio', color='#24476B', linewidth=2.2)
    axes[0].plot(epochs, [row['beta_visual'] / 0.75 for row in rows],
                 label='Visual', color='#E76F51', linewidth=2.2)
    axes[0].set_title('Batch-level gate activation')
    axes[0].set_ylabel('Ratio')
    axes[0].legend(loc='upper right')

    axes[1].plot(
        epochs, [row['sample_useful_ratio_audio'] for row in rows],
        label='Audio', color='#24476B', linewidth=2.2
    )
    axes[1].plot(
        epochs, [row['sample_useful_ratio_visual'] for row in rows],
        label='Visual', color='#E76F51', linewidth=2.2
    )
    axes[1].set_title('Sample-level useful ratio')
    axes[1].legend(loc='upper right')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xlim(0, 99)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis='y', color='#D8D8D8', linewidth=0.6, alpha=0.7)
    fig.suptitle('Selector Dynamics of the Balanced Symmetric Run',
                 y=1.02, fontsize=12, fontweight='semibold')
    fig.tight_layout()
    save_figure(fig, output_dir, 'selector_gate_best_run')


def plot_selector_dynamics_multiseed(result_dirs, output_dir, output_name='selector_dynamics_multiseed.png'):
    aggregated = aggregate_selector_runs(result_dirs)
    if aggregated is None:
        return

    epochs = aggregated['epoch']

    def draw_mean_std(ax, mean_key, std_key, label, color, linestyle='-'):
        mean = aggregated[mean_key]
        std = aggregated[std_key]
        ax.plot(epochs, mean, label=label, color=color, linestyle=linestyle, linewidth=2.2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    plt.figure(figsize=(9.5, 4.8))
    ax = plt.gca()
    draw_mean_std(ax, 'sim_audio_mean', 'sim_audio_std', 'Audio Similarity Mean±Std', '#1d3557')
    draw_mean_std(ax, 'sim_visual_mean', 'sim_visual_std', 'Visual Similarity Mean±Std', '#e76f51')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Selector Similarity Dynamics Across Seeds')
    plt.ylim(bottom=0.0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selector_similarity_multiseed.png'), dpi=220)
    plt.close()

    plt.figure(figsize=(9.5, 5.8))
    ax = plt.gca()
    draw_mean_std(ax, 'beta_audio_mean', 'beta_audio_std', 'Audio Beta Mean±Std', '#1d3557')
    draw_mean_std(ax, 'beta_visual_mean', 'beta_visual_std', 'Visual Beta Mean±Std', '#e76f51')
    draw_mean_std(
        ax, 'sample_useful_ratio_audio_mean', 'sample_useful_ratio_audio_std',
        'Audio Useful Ratio Mean±Std', '#457b9d', linestyle='--'
    )
    draw_mean_std(
        ax, 'sample_useful_ratio_visual_mean', 'sample_useful_ratio_visual_std',
        'Visual Useful Ratio Mean±Std', '#f4a261', linestyle='--'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Gate / Useful Ratio')
    plt.title('Selector Gate Dynamics Across Seeds', pad=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.18), columnspacing=1.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, 'selector_gate_multiseed.png'), dpi=220)
    plt.close()


def plot_metric_tradeoff(records, output_dir):
    plt.figure(figsize=(7, 5))
    for record in records:
        plt.scatter(record['best_checkpoint_audio'] * 100.0, record['best_checkpoint_visual'] * 100.0,
                    s=80, label=record['name'])
        plt.text(record['best_checkpoint_audio'] * 100.0 + 0.03,
                 record['best_checkpoint_visual'] * 100.0 + 0.03,
                 record['tau'] + '/' + record['lambda'], fontsize=8)
    plt.axvline(PAPER_RESULT['audio'] * 100.0, color='#1d3557', linestyle='--', label='Paper Audio')
    plt.axhline(PAPER_RESULT['visual'] * 100.0, color='#e76f51', linestyle='--', label='Paper Visual')
    plt.xlabel('Audio Accuracy at Best Fusion Checkpoint (%)')
    plt.ylabel('Visual Accuracy at Best Fusion Checkpoint (%)')
    plt.title('Audio-Visual Tradeoff at Best Fusion Checkpoints')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'audio_visual_tradeoff.png'), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='./results', type=str)
    parser.add_argument('--output_dir', default='./results/report_figures', type=str)
    parser.add_argument('--aggregate_dirs', nargs='*', default=None,
                        help='optional list of result dirs for multi-seed selector plotting')
    parser.add_argument('--best_run_dir', default=None, type=str,
                        help='explicit result dir used for all single-run figures')
    args = parser.parse_args()

    configure_plot_style()
    ensure_dir(args.output_dir)
    records = build_run_records(args.results_root)
    if not records:
        raise RuntimeError('No run summaries were found under {}'.format(args.results_root))

    best_record = records[0]
    if args.best_run_dir:
        requested_dir = os.path.abspath(args.best_run_dir)
        matching = [
            record for record in records
            if os.path.abspath(record['result_dir']) == requested_dir
        ]
        if not matching:
            raise RuntimeError('Best-run result directory was not found: {}'.format(args.best_run_dir))
        best_record = matching[0]
    plot_ablation_best_fusion(records, args.output_dir)
    plot_best_checkpoint_compare(best_record, args.output_dir)
    plot_best_run_accuracy_curves(best_record, args.output_dir)
    plot_best_run_training_loss(best_record, args.output_dir)
    plot_fusion_curves(records, args.output_dir)
    plot_selector_dynamics(best_record, args.output_dir)
    plot_metric_tradeoff(records, args.output_dir)
    if args.aggregate_dirs:
        plot_selector_dynamics_multiseed(args.aggregate_dirs, args.output_dir)
        plot_training_loss_multiseed(args.aggregate_dirs, args.output_dir)
        plot_selector_similarity_threshold_multiseed(args.aggregate_dirs, args.output_dir)
        plot_selector_activation_multiseed(args.aggregate_dirs, args.output_dir)
        plot_asymmetry_evidence_multiseed(args.aggregate_dirs, args.output_dir)
        plot_multiseed_performance_vs_paper(args.aggregate_dirs, args.output_dir)

    print('Saved figures to {}'.format(args.output_dir))
    for file_name in sorted(os.listdir(args.output_dir)):
        print(file_name)


if __name__ == '__main__':
    main()
