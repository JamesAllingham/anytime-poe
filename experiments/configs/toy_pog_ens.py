from math import ceil

from ml_collections import config_dict

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    # config.dataset_name = 'gen_matern'
    # config.dataset = config_dict.ConfigDict()
    # config.dataset.n_samples = 1200
    # config.dataset.random_seed = 42
    # config.dataset.noise_std = 0.15
    config.dataset_name = 'gen_simple_1d'
    config.dataset = config_dict.ConfigDict()
    config.dataset.n_samples = 692
    config.dataset.random_seed = 42
    config.dataset.noise_std = 0.25
    config.dataset.heteroscedastic = False

    config.batch_size = 500
    config.epochs = 201

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 1e-4

    config.β_schedule = config_dict.ConfigDict()
    config.β_schedule.name = 'sigmoid'
    config.β_schedule.start = 2.
    config.β_schedule.end = 32.
    num_train = int(int(config.dataset.n_samples * 0.8) * 0.9)
    num_batches_per_epoch = ceil(num_train / config.batch_size)
    config.β_schedule.steps = config.epochs * num_batches_per_epoch
    # config.β = 1.

    config.model_name = 'PoG_Ens'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False
    config.model.noise = 'homo'

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 2
    config.model.net.hidden_size = 50
    out_size = 1
    out_size_mult = 2 if config.model.noise == 'hetero' else 1
    config.model.net.out_size = out_size * out_size_mult

    return config
