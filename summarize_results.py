import argparse
import os

from utils.result_summary import write_run_summary


def infer_paths(result_dir):
    accuracy_candidates = []
    selector_candidates = []

    for file_name in os.listdir(result_dir):
        if file_name.endswith('_selector.csv'):
            selector_candidates.append(file_name)
        elif file_name.endswith('.csv') and file_name != 'run_summary.csv':
            accuracy_candidates.append(file_name)

    accuracy_candidates.sort()
    selector_candidates.sort()

    accuracy_path = os.path.join(result_dir, accuracy_candidates[0]) if accuracy_candidates else None
    selector_path = os.path.join(result_dir, selector_candidates[0]) if selector_candidates else None
    return accuracy_path, selector_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True, type=str)
    parser.add_argument('--dataset', default='N/A', type=str)
    parser.add_argument('--modality', default='N/A', type=str)
    parser.add_argument('--grad_strategy', default='N/A', type=str)
    parser.add_argument('--alpha', default='N/A', type=str)
    parser.add_argument('--selector_tau', default='N/A', type=str)
    parser.add_argument('--selector_lambda', default='N/A', type=str)
    parser.add_argument('--selector_tau_audio', default='N/A', type=str)
    parser.add_argument('--selector_tau_visual', default='N/A', type=str)
    parser.add_argument('--selector_lambda_audio', default='N/A', type=str)
    parser.add_argument('--selector_lambda_visual', default='N/A', type=str)
    parser.add_argument('--selector_start_epoch', default='N/A', type=str)
    args = parser.parse_args()

    accuracy_path, selector_path = infer_paths(args.result_dir)
    if accuracy_path is None:
        raise FileNotFoundError('No accuracy csv was found in {}'.format(args.result_dir))

    meta = {
        'dataset': args.dataset,
        'modality': args.modality,
        'grad_strategy': args.grad_strategy,
        'alpha': args.alpha,
        'selector_tau': args.selector_tau,
        'selector_lambda': args.selector_lambda,
        'selector_tau_audio': args.selector_tau_audio,
        'selector_tau_visual': args.selector_tau_visual,
        'selector_lambda_audio': args.selector_lambda_audio,
        'selector_lambda_visual': args.selector_lambda_visual,
        'selector_start_epoch': args.selector_start_epoch,
    }
    summary_text, text_path, csv_path = write_run_summary(args.result_dir, accuracy_path, selector_path, meta)
    print(summary_text)
    print('Summary files saved to:')
    print(text_path)
    print(csv_path)


if __name__ == '__main__':
    main()
