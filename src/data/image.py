"""Image dataset functionality."""
from typing import Callable, Tuple, Union
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, datasets


DATASET_MEAN = {
    'MNIST': (0.1307,),
    'FashionMNIST': (0.2860,),
    'KMNIST': (0.1918,),
    'SVHN': (0.4377, 0.4438, 0.4728),
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4866, 0.4409),
}


DATASET_STD = {
    'MNIST': (0.3081,),
    'FashionMNIST': (0.3530,),
    'KMNIST': (0.3483,),
    'SVHN': (0.1980, 0.2010, 0.1970),
    'CIFAR10': (0.2470, 0.2435, 0.2616),
    'CIFAR100': (0.2673, 0.2564, 0.2762),
}


class Flatten:
    """Transform to flatten an image for use with MLPs."""
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.ravel(array)


class MoveChannelDim:
    """Transform to change from PyTorch image ordering to Jax/TF ordering."""
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.moveaxis(array, 0, -1)


class ToNumpy:
    """Transform to convert from a PyTorch Tensor to a Numpy ndarray."""
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return np.array(tensor, dtype=np.float32)


def get_image_dataset(
    dataset_name: str,
    data_dir: str = '../raw_data',
    flatten_img: bool = False,
    val_percent: float = 0.1,
    random_seed: int = 42,
    train_augmentations: list[Callable] = [],
    test_augmentations: list[Callable] = [],
) -> Union[Tuple[data.Dataset, data.Dataset], Tuple[data.Dataset, data.Dataset, data.Dataset]]:
    """Provides PyTorch `Dataset`s for the specified image dataset_name.

    Args:
        dataset_name: the `str` name of the dataset. E.g. `'MNIST'`.

        data_dir: the `str` directory where the datasets should be downloaded to and loaded from. (Default: `'../raw_data'`)

        flatten_img: a `bool` indicating whether images should be flattened. (Default: `False`)

        val_percent: the `float` percentage of training data to use for validation. (Default: `0.1`)

        random_seed: the `int` random seed for splitting the val data and applying random affine transformations. (Default: 42)

        train_augmentations: a `list` of augmentations to apply to the training data. (Default: `[]`)

        test_augmentations: a `list` of augmentations to apply to the test data. (Default: `[]`)

        η_low, η_high: optional `Array`s controlling the affine transformations to apply to the data.
        To be concrete, these arrays represent the lower and upper bounds of a uniform distribution
        from which affine transformation parameters are drawn.
        For more information see `transformations.affine.gen_transform_mat`.
        (Default: `None`)

    Returns:
        `(train_dataset, test_dataset)` if `val_percent` is 0 otherwise `(train_dataset, test_dataset, val_dataset)`
    """
    dataset_choices = ['MNIST', 'FashionMNIST', 'KMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    if dataset_name not in dataset_choices:
        msg = f"Dataset should be one of {dataset_choices} but was {dataset_name} instead."
        raise RuntimeError(msg)

    if dataset_name == 'MNIST':
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    elif dataset_name == 'FashionMNIST':
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    elif dataset_name == 'KMNIST':
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    elif dataset_name == 'SVHN':
        train_kwargs = {"split": 'train'}
        test_kwargs = {"split": 'test'}

    elif dataset_name == 'CIFAR10':
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    else:
        assert dataset_name == 'CIFAR100'

        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    common_transforms = [
        ToNumpy(),
        MoveChannelDim(),
    ]
    if flatten_img:
        common_transforms += [Flatten()]

    transform_train = transforms.Compose(
        train_augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(DATASET_MEAN[dataset_name], DATASET_STD[dataset_name]),
        ] + common_transforms
    )

    transform_test = transforms.Compose(
        test_augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(DATASET_MEAN[dataset_name], DATASET_STD[dataset_name]),
        ] + common_transforms
    )

    dataset = getattr(datasets, dataset_name)
    train_dataset = dataset(
        **train_kwargs, transform=transform_train, download=True, root=data_dir,
    )
    test_dataset = dataset(
        **test_kwargs, transform=transform_test, download=True, root=data_dir,
    )

    if val_percent != 0.:
        num_train, num_val = train_val_split_sizes(len(train_dataset), val_percent)

        train_dataset, val_dataset = data.random_split(
            train_dataset, [num_train, num_val],
            torch.Generator().manual_seed(random_seed) if random_seed is not None else None
        )

        return train_dataset, test_dataset, val_dataset
    else:
        return train_dataset, test_dataset


def train_val_split_sizes(num_train: int, val_percent: float) -> Tuple[int, int]:
    num_val = int(val_percent * num_train)
    num_train = num_train - num_val

    return num_train, num_val
