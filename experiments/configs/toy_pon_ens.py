from ml_collections import config_dict

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_name = 'gen_matern'
    config.dataset = config_dict.ConfigDict()
    config.dataset.n_samples = 1200
    config.dataset.random_seed = 42
    config.dataset.noise_std = 0.15

    config.batch_size = 256
    config.epochs = 50

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 1e-4

    config.model_name = 'PoN_Ens'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False
    config.model.noise = 'homo'

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 2
    config.model.net.hidden_size = 100
    out_size = 1
    out_size_mult = 2 if config.model.noise == 'hetero' else 1
    config.model.net.out_size = out_size * out_size_mult

    return config
