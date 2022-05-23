from math import ceil

from ml_collections import config_dict

from src.data import METADATA, train_val_split_sizes

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_name = 'MNIST'
    config.val_percent = 0.1
    config.batch_size = 256
    config.epochs = 50

    config.optim_name = 'sgdw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.optim.momentum = 0.9
    config.learning_rate = 1e-4

    config.model_name = 'Cls_Ens'
    config.model = config_dict.ConfigDict()
    config.model.size = 5
    config.model.learn_weights = False

    config.model.net = config_dict.ConfigDict()
    config.model.net.depth = 2
    config.model.net.hidden_size = 100
    config.model.net.out_size = 10

    return config
