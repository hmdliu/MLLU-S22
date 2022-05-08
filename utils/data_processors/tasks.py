
import abc
import torch
import logging
import datasets
import functools

from collections import OrderedDict
from typing import Callable, List, Mapping

from utils.metrics import metrics
from utils.trainers.trainer_utils import pad_punctuation

logger = logging.getLogger(__name__)

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = {
        'train': 'train',
        'validation': 'validation', 
        'test': 'test'
    }

    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool=False,
                       prefix: str=None,
                       extra_fields={}):
        src_prefix = self.name if prefix is None else prefix
        sources = [src_prefix]+sources if add_prefix else sources
        return {
            'source': ' '.join(sources),
            'target': ' '.join(targets),
            'task': self.name
        }

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs
   
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == 'validation':
            return indices[:validation_size]
        else:
            return indices[validation_size:]
        
    def map_dataset(self, dataset, add_prefix):
        # print('dataset size:', len(dataset))
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False):
        if self.name == 'squad' and split != 'train':
            dataset = self.load_dataset(split='validation')
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        elif self.name == 'yelp' and split != 'test':
            dataset = self.load_dataset(split='train')
            indices = self.get_split_indices(split, dataset, validation_size=38000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return self.map_dataset(dataset, add_prefix)    

class Squad(AbstractTask):
    name = 'squad'
    metric = [metrics.squad]
    metric_names = ['squad']

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = [
            'question:', question,
            'context:', context
        ]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)

class MNLI(AbstractTask):
    name = 'mnli'
    labels_list = ['0', '1', '2']
    split_to_data_split = {
        'train': 'train',
        'validation': 'validation_mismatched',
        'test': 'validation_matched'
    }
    metric = [metrics.accuracy]
    metric_names = ['accuracy']

    def load_dataset(self, split):
        if split == 'train':
            return datasets.load_dataset('glue', 'mnli', split='train')
        print('Concatenating MNLI-matched and MNLI-mismatched ...')
        matched = datasets.load_dataset('glue', 'mnli', split='validation_matched')
        mismatched = datasets.load_dataset('glue', 'mnli', split='validation_mismatched')
        return datasets.concatenate_datasets([matched, mismatched])

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            'premise:', example['premise'],
            'hypothesis:', example['hypothesis']
        ]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class RACE(AbstractTask):
    name = 'race'
    labels_list = ['A', 'B', 'C', 'D']
    metric = [metrics.accuracy]
    metric_names = ['accuracy']

    def load_dataset(self, split):
        return datasets.load_dataset('race', 'all', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            'question:', example['question'],
            'options:',
            'A', example['options'][0],
            'B', example['options'][1],
            'C', example['options'][2],
            'D', example['options'][3],
            'article:', example['article'],
        ]
        tgt_texts = [example['answer']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class Yelp(AbstractTask):
    name = 'yelp'
    labels_list = ['0', '1']
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ['accuracy']

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ['review:', example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mnli', MNLI),
        ('race', RACE),
        ('yelp', Yelp),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
