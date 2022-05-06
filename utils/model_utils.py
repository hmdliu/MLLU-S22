
import random
from ray import tune
from transformers import AutoModelForSeq2SeqLM

def model_init_func(config, model_args, delta_args, tokenizer_size=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=False
    )
    if tokenizer_size is not None:
        model.resize_token_embeddings(tokenizer_size)
    if delta_args.delta_type.lower() != 'none':
        from opendelta import AutoDeltaConfig, AutoDeltaModel
        delta_config = AutoDeltaConfig.from_dict(vars(delta_args))
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict=True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    # print(my_model)
    return model

def get_hp_space_func(dataset):

    bs1, bs2 = 16, 32
    if dataset[0] in ('race', 'squad'):
        bs1, bs2 = 4, 8

    def hp_space(trial):
        return {
            'max_steps': tune.choice([10000, 20000, 40000]),
            'learning_rate': tune.loguniform(1e-5, 1e-3),
            'per_device_train_batch_size': tune.choice([bs1, bs2]),
            'per_device_eval_batch_size': bs2,
        }

    return hp_space
