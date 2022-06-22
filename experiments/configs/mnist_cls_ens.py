from math import ceil

from ml_collections import config_dict

from src.data import METADATA, train_val_split_sizes

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_name = 'MNIST'
    config.val_percent = 0.1
    config.batch_size = 512
    config.epochs = 50

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 3e-3

    num_train, _ = train_val_split_sizes(METADATA['num_train'][config.dataset_name], config.val_percent)
    num_batches_per_epoch = ceil(num_train / config.batch_size)

    config.lr_schedule_name = 'warmup_cosine_decay_schedule'
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 3 * config.learning_rate
    config.lr_schedule.end_value = 1/10 * config.learning_rate
    config.lr_schedule.decay_steps = config.epochs * num_batches_per_epoch
    config.lr_schedule.warmup_steps = int(0.2 * config.lr_schedule.decay_steps)

    config.model_name = 'Cls_Ens'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 5
    config.model.net.hidden_size = 200
    config.model.net.out_size = 10
    config.model.net.p_drop = .1

    return config
