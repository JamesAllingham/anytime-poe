from math import ceil

from ml_collections import config_dict

from src.data import METADATA, train_val_split_sizes

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.n_classes = 4

    config.dataset_name = 'gen_spirals'
    config.dataset = config_dict.ConfigDict()
    config.dataset.n_samples = 6000
    config.dataset.random_seed = 42
    config.dataset.noise_std = 0.1
    config.dataset.n_arms = config.n_classes
    config.dataset.start_angle = 90
    config.dataset.stop_angle = 180

    config.batch_size = 256
    config.epochs = 501

    # config.train_data_noise = 0.05

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 1e-4

    config.model_name = 'Hard_OvR_Ens'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False

    num_train = int(int(config.dataset.n_samples * 0.8) * 0.9)
    num_batches_per_epoch = ceil(num_train / config.batch_size)

    # config.lr_schedule_name = 'warmup_cosine_decay_schedule'
    # config.lr_schedule = config_dict.ConfigDict()
    # config.lr_schedule.peak_value = config.learning_rate
    # config.lr_schedule.end_value = 1/3 * config.learning_rate
    # config.lr_schedule.decay_steps = config.epochs * num_batches_per_epoch
    # config.lr_schedule.warmup_steps = int(0.01 * config.lr_schedule.decay_steps)

    config.β_schedule = config_dict.ConfigDict()
    config.β_schedule.name = 'linear'
    config.β_schedule.start = 1.
    config.β_schedule.end = 1.5
    # config.β_schedule.steps = int(config.epochs * num_batches_per_epoch * 0.9)
    config.β_schedule.steps = config.epochs * num_batches_per_epoch
    # config.β = 1.

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 5
    config.model.net.hidden_size = 100
    config.model.net.out_size = config.n_classes

    return config
