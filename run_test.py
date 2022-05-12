# ---------------------------------------------------------------
# Seq-to-Seq Delta Tuning Methods Evaluation (T5-Base)
# References:
#  - Hugging Face: https://github.com/huggingface/transformers
#  - OpenDelta: https://github.com/thunlp/OpenDelta
# ---------------------------------------------------------------

import os
import sys
import json
import random
import logging
import functools

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed,
)
from datasets import concatenate_datasets
from transformers.trainer_utils import is_main_process

from utils.config import ConfigParser, get_test_config, save_delta_config
from utils.seq2seq_trainer import Seq2SeqTrainer
from utils.data_processors import AutoTask, TaskDataCollatorForSeq2Seq, AutoPostProcessor
from utils.trainers.model_args import ModelArguments
from utils.trainers.trainer_args import TrainingArguments, DataTrainingArguments

logger = logging.getLogger(__name__)
TASK_TO_METRICS = {
    'squad': ['em', 'f1'],
    'mnli': ['accuracy'],
    'race': ['accuracy'],
    'yelp': ['accuracy']
}

def main():

    assert len(sys.argv) == 4, 'Usage: python run_seq2seq.py [dataset] [delta_type] [data_ratio]'

    # init parser & arguments
    parser = ConfigParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    addict_config = get_test_config(dataset=sys.argv[1], delta_type=sys.argv[2], data_ratio=float(sys.argv[3]))
    model_args, data_args, training_args, delta_args = parser.parse_addict(addict_config)
    # print(model_args, data_args, training_args, delta_args)

    # setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # log summary on each process:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    # set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info('Training/evaluation parameters %s', training_args)

    # set seed before initializing model.
    set_seed(training_args.seed)

    # load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(f'\n[max length]: tokenizer={tokenizer.model_max_length}, config={data_args.max_source_length}\n')

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if delta_args.delta_type.lower() != 'none':
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        delta_config = AutoDeltaConfig.from_dict(vars(delta_args))
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict=True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

    # init dataset
    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # temporarily set max_target_length for training
    padding = 'max_length' if data_args.pad_to_max_length else False
    def preprocess_function(examples, max_target_length):
        model_inputs = tokenizer([s for s in examples['source']], max_length=data_args.max_source_length, padding=padding, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples['target']], max_length=max_target_length, padding=padding, truncation=True)
        if padding == 'max_length' and data_args.ignore_pad_token_for_loss:
            # replace pad_token_id with -100 to exclude from the loss computation
            labels['input_ids'] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    column_names = ['source', 'target']
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [
            AutoTask.get(dataset_name, dataset_config_name, seed=data_args.data_seed).get(
                split='train',
                split_validation_test=training_args.split_validation_test,
                add_prefix=True,
                n_obs=data_args.max_train_samples
            )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)
        ]
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer,
                default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)
        ]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)
   
    if training_args.do_eval:
        eval_datasets = {
            eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config, seed=data_args.data_seed).get(
                split='validation', 
                split_validation_test=training_args.split_validation_test,
                add_prefix=True,
                n_obs=data_args.max_val_samples
            )
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)
        }
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer,
                default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)
        ]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_test:
        test_datasets = {
            test_dataset: AutoTask.get(test_dataset, test_dataset_config, seed=data_args.data_seed).get(
                split='test',
                split_validation_test=training_args.split_validation_test,
                add_prefix=True,
                n_obs=data_args.max_test_samples
            )
            for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)
        }
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer,
                default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)
        ]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    eval_metrics = [
        AutoTask.get(dataset_name, dataset_config_name).metric
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)
    ][0]

    data_info = None
    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer, data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result
    print(f'\n[metrics]: {TASK_TO_METRICS[data_args.dataset_name[0]]}\n')

    # init Seq2Seq Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        delta_args=delta_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        data_info=data_info,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        evaluation_metrics=TASK_TO_METRICS[data_args.dataset_name[0]],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]    # enlarge patience when eval interval is small
    )

    # save exp config
    if trainer.is_world_process_zero():
       save_delta_config(addict_config, training_args.output_dir)

    if training_args.do_train:

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # load hyperparameters from the best run 
        with open(delta_args.hp_path, 'r') as f:
            best_run = json.load(f)
        for n, v in best_run.items():
            setattr(trainer.args, n, v)
        print('[Config Path]:', delta_args.hp_path)
        print('[Best Config]:', best_run)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print('[Train Result]:', train_result)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end) / (1000*60)
            performance_metrics.update({'total_time in minutes': total_time})

        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics['train_samples'] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics('train', train_metrics)
        trainer.save_metrics('train', train_metrics)
        trainer.save_state()
    
    # evaluation
    results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(
                eval_dataset=eval_dataset,
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams,
            )
            trainer.log_metrics('eval', metrics)
            trainer.save_metrics('eval', metrics)
        results['evaluate'] = metrics

    # test
    if training_args.do_test:
        logger.info('*** Test ***')
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(
                eval_dataset=test_dataset,
                max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
                metric_key_prefix='test'
            )
            trainer.log_metrics('test', metrics)
            trainer.save_metrics('test', metrics)
        results['test'] = metrics

    # dump results
    with open(f'./test/results.jsonl', 'a') as f:
        string = json.dumps(results, indent=4, sort_keys=True)
        f.write(f'{delta_args.hp_path}\n{string}\n\n')

if __name__ == '__main__':
    main()
    
    
