
import os
import sys
import json

DATASET_LIST = ['mnli', 'race', 'squad', 'yelp']
DATA_RATIO_LIST = ['1.000', '0.100', '0.010', '0.001']
DELTA_TYPE_LIST = ['none', 'adapter', 'bitfit', 'lora', 'prefix']
grep_between = lambda s, pre, post: s[s.find(pre)+len(pre):s.find(post)]

def dump_exp(search_dir, output_dir, dataset, delta_type, data_ratio):

    # formulate current directory
    curr_dir = os.path.join(search_dir, dataset, delta_type, data_ratio, 'hps_log')
    if not os.path.isdir(curr_dir):
        return
    print('\n[Curr Dir]:', curr_dir)

    # list available trials
    trials = [d for d in os.listdir(curr_dir) if d.find('_objective') != -1]
    print('[Num Trials]:', len(trials))
    if len(trials) < 8:
        print('[Info]: Hyper Search is still running.')
        return

    # grep results
    results, metrics, paths = [], [], []
    for i in range(len(trials)):
        res = os.path.join(curr_dir, trials[i], 'stdout')
        try:
            with open(res, 'r') as f:
                lines = f.read().split('\n')
            filtered_lines = [l for l in lines if l.startswith("{'eval_loss':")]
            results.append(eval(max(filtered_lines, key=lambda l: eval(l)['eval_average_metrics'])))
            metrics.append('\n'.join(filtered_lines))
            paths.append(res)
        except:
            print(f'Skip trial {i}')

    # print best results
    best_results = max(results, key=lambda d: d['eval_average_metrics'])
    print(f"[Best Ave Metrics]: {best_results['eval_average_metrics']:.2f}")
    if curr_dir.find('squad') != -1:
        print(f"[Best EM]: {best_results['eval_em']:.2f}")
        print(f"[Best F1]: {best_results['eval_f1']:.2f}")

    # dump best config
    best_path = paths[results.index(best_results)]
    best_config = {
        'learning_rate': float(grep_between(best_path, 'learning_rate=', ',max_steps')),
        'max_steps': int(grep_between(best_path, 'max_steps=', ',per_device_train_batch_size')),
        'per_device_train_batch_size': int(grep_between(best_path, 'per_device_train_batch_size=', '_2022')),
    }
    best_config['per_device_eval_batch_size'] = 8 if best_config['per_device_train_batch_size'] in (4, 8) else 32
    print('[Best Config]:', best_config)
    config_path = os.path.join(output_dir, f'{dataset}_{delta_type}_{data_ratio}.json')
    print('[Config Path]:', config_path)
    with open(config_path, 'w') as f:
        f.write(json.dumps(best_config, indent=4, sort_keys=True))

    # dump convergence log
    log_path = os.path.join(output_dir, f'{dataset}_{delta_type}_{data_ratio}.log')
    print('[Log Path]:', log_path)
    with open(log_path, 'w') as f:
        f.write(metrics[results.index(best_results)])

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python dump_results.py [search_dir] [output_dir]'
    assert os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]), 'Invalid path(s)'
    search_dir, output_dir = sys.argv[1], sys.argv[2]
    print('[Search Dir]:', search_dir)
    print('[Output Dir]:', output_dir)
    
    # dump results of all experiments
    for dataset in DATASET_LIST:
        for delta_type in DELTA_TYPE_LIST:
            for data_ratio in DATA_RATIO_LIST:
                dump_exp(search_dir, output_dir, dataset, delta_type, data_ratio)
