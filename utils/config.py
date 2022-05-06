
import os
import json
import argparse
import dataclasses

from copy import deepcopy
from addict import Dict
from pathlib import Path
from transformers import HfArgumentParser

BASE_CONFIG = Dict({
    'seed': 42,
    'do_train': True,
    'do_eval': True,
    'do_test': False,
    'split_validation_test': True,
    'dataset_config_name': ['en'],
    'eval_dataset_config_name': ['en'],
    'test_dataset_config_name': ['en'],
    'max_source_length': 128,
    'metric_for_best_model': 'average_metrics',
    'per_device_eval_batch_size': 32,
    'predict_with_generate': True,
    'greater_is_better': True,
    'load_best_model_at_end': True,
    'eval_steps': 2000,
    'save_steps': 2000,
    'warmup_steps': 0,
    'logging_steps': 2000,
    'evaluation_strategy': 'steps',
    'model_name_or_path': 'google/t5-base-lm-adapt',
    'tokenizer_name': 'google/t5-base-lm-adapt',
    'overwrite_output_dir': True,
    'save_strategy': 'steps',
    'save_total_limit': 1,
    'disable_tqdm': True,
    'compute_time': True,
    'push_to_hub': False,
})

TRAIN_SIZE = {
    'squad': 87599,
    'race': 87866,
    'mnli': 392702,
    'yelp': 559000
}

def get_search_config(dataset: str, delta_type: str, data_ratio: float) -> Dict:

    assert dataset in ('squad', 'race', 'mnli', 'yelp')
    assert delta_type in ('none', 'adapter', 'bitfit', 'lora', 'prefix')
    # assert 0 <= data_ratio <= 1

    # init config
    config = deepcopy(BASE_CONFIG)
    config.task_name = dataset
    config.eval_dataset_name = dataset
    config.test_dataset_name = dataset
    config.delta_type = delta_type
    config.max_train_samples = int(data_ratio * TRAIN_SIZE[dataset])
    config.output_dir = os.path.abspath(f'./outputs/{dataset}/{delta_type}/{data_ratio:.3f}')
    config = update_dataset_config(config)
    config = update_delta_config(config)
    os.makedirs(config.output_dir, exist_ok=True)

    # test run
    if data_ratio == -1:
        config.disable_tqdm = False
        config.max_steps = 100
        config.eval_steps = 50
        config.max_train_samples = 160
        config.max_val_samples = 160

    return config

def get_eval_config(dataset: str, delta_type: str, data_ratio: float) -> Dict:
    raise NotImplementedError

def update_dataset_config(config):
    if config.task_name in ('squad', 'race'):
        config.max_source_length = 512
    return config

def update_delta_config(config):
    if config.delta_type in ('none', 'bitfit'):
        pass
    elif config.delta_type == 'prefix':
        config.unfrozen_modules = ['deltas']
    elif config.delta_type == 'lora':
        config.lora_r = 8
        config.unfrozen_modules = ['deltas', 'layer_norm', 'final_layer_norm']
    elif config.delta_type == 'adapter':
        config.bottleneck_dim = 24
        config.unfrozen_modules = ['deltas', 'layer_norm', 'final_layer_norm']
    else:
        raise ValueError(f'Invalid delta type: {config.delta_type}.')
    return config

def save_delta_config(config_dict, output_dir):
    with open(os.path.join(output_dir, 'delta_config.json'), 'w') as f:
        json.dump(config_dict, f)

class ConfigParser(HfArgumentParser):

    def parse_addict(self, data: Dict, return_remaining_args=True):
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        return (*outputs, remain_args) if return_remaining_args else (*outputs,)

    def parse_json(self, json_file: str, return_remaining_args=True):
        outputs = []
        data = json.loads(Path(json_file).read_text())
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        return (*outputs, remain_args) if return_remaining_args else (*outputs,)

if __name__ == '__main__':
    test_config = get_config(
        dataset='mnli',
        delta_type='adapter',
        data_ratio=0.1
    )
    for k, v in test_config.items():
        print(f'[{k}]: {v}')
