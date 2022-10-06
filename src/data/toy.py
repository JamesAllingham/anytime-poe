import math
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split


def gen_matern(
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = 42,
    noise_std: Optional[float] = None,
):
    if n_samples is None:
        n_samples = 1200
    if random_seed is not None:
        np.random.seed(random_seed)
    if noise_std is None:
        noise_std = 0.15

    def _gen_matern():
        from GPy.kern.src.sde_matern import Matern32

        lengthscale = 0.5
        variance = 1.0
        sig_noise = noise_std

        x1 = np.random.uniform(-2, -1, n_samples//2)[:, np.newaxis]
        x2 = np.random.uniform(0.5, 2.5, n_samples//2)[:, np.newaxis]
        x = np.concatenate([x1, x2])

        n_points = (n_samples//2)*2
        x = np.concatenate([x1, x2], axis=0)
        x.sort(axis=0)

        k = Matern32(input_dim=1, variance=variance, lengthscale=lengthscale)
        C = k.K(x, x) + np.eye(n_points) * sig_noise ** 2

        y = np.random.multivariate_normal(np.zeros((n_points)), C)[:, np.newaxis]

        return x.astype(np.float32), y.astype(np.float32)

    x, y = _gen_matern()

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(x, y, random_seed)

    (x_train, x_test, x_valid) = _normalise_by_train(x_train, x_test, x_valid)
    (y_train, y_test, y_valid) = _normalise_by_train(y_train, y_test, y_valid)

    return _zip_dataset(x_train, y_train), _zip_dataset(x_test, y_test), _zip_dataset(x_valid, y_valid)


def gen_simple_1d(
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = 42,
    noise_std: Optional[float] = None,
    heteroscedastic: bool = False,
):
    if n_samples is None:
        n_samples = 1503
    if random_seed is not None:
        np.random.seed(random_seed)
    if noise_std is None:
        noise_std = 0.25

    def _gen_simple_1d():
        x0 = np.random.uniform(-1, 0, size=int(n_samples / 3))
        x1 = np.random.uniform(1.7, 2.5, size=int(n_samples / 3))
        x2 = np.random.uniform(4, 5, size=int(n_samples / 3))
        x = np.concatenate([x0, x1, x2])

        def function(x):
            return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

        y = function(x)

        if heteroscedastic:
            noise = np.random.randn(*x.shape) * np.abs(noise_std * np.abs(x))
        else:
            noise = np.random.randn(*x.shape) * noise_std

        y = y + noise

        x = x[:, np.newaxis]
        y = y[:, np.newaxis]

        return x.astype(np.float32), y.astype(np.float32)

    x, y = _gen_simple_1d()

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(x, y, random_seed)

    (x_train, x_test, x_valid) = _normalise_by_train(x_train, x_test, x_valid)
    (y_train, y_test, y_valid) = _normalise_by_train(y_train, y_test, y_valid)

    return _zip_dataset(x_train, y_train), _zip_dataset(x_test, y_test), _zip_dataset(x_valid, y_valid)


def gen_wiggle(
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = 42,
    noise_std: Optional[float] = None,
):
    if n_samples is None:
        n_samples = 900
    if random_seed is not None:
        np.random.seed(random_seed)
    if noise_std is None:
        noise_std = 0.25

    x = np.random.randn(n_samples) * 2.5 + 5

    def function(x):
        return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

    y = function(x)

    noise = np.random.randn(*x.shape) * noise_std
    y = y + noise

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(x, y, random_seed)

    (x_train, x_test, x_valid) = _normalise_by_train(x_train, x_test, x_valid)
    (y_train, y_test, y_valid) = _normalise_by_train(y_train, y_test, y_valid)

    return _zip_dataset(x_train, y_train), _zip_dataset(x_test, y_test), _zip_dataset(x_valid, y_valid)


def gen_spirals(
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = 1234,
    noise_std: Optional[float] = None,
    n_arms: int = 2,
    start_angle: int = 0,
    stop_angle: int = 360
):
    """Adapted from: https://github.com/DatCorno/N-Arm-Spiral-Dataset

    n_samples: The total number of points generated.
    noise_std: Standard deviation of Gaussian noise added to the data.
    random_seed: Determines random number generation for dataset shuffling and noise.
    n_arms: The number of arms in the spiral.
    start_angle: The starting angle for each spiral arm (in degrees).
    stop_angle: The stopping angle for each spiral arm (in degrees).
    """
    if n_samples is None:
        n_samples = 300
    if random_seed is not None:
        np.random.seed(random_seed)
    if noise_std is None:
        noise_std = 0.1

    n_samples = math.floor(n_samples / n_arms)
    x = np.empty((0, 2))
    y = np.empty((0,))

    def _generate_spiral(n_samples, start_angle, stop_angle, angle, noise_std, random_seed=1234):
        # Generate points from the square root of random data inside an uniform distribution on [0, 1).
        points = math.radians(start_angle) + np.sqrt(np.random.rand(n_samples, 1)) * math.radians(stop_angle)

        # Apply a rotation to the points.
        rotated_x_axis = np.cos(points) * points + np.random.rand(n_samples, 1) * noise_std
        rotated_y_axis = np.sin(points) * points + np.random.rand(n_samples, 1) * noise_std

        # Stack the vectors inside a samples x 2 matrix.
        rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))

        def _rotate_point(point, angle):
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_point = rotation_matrix.dot(point)
            return rotated_point

        return np.apply_along_axis(_rotate_point, 1, rotated_points, math.radians(angle))

    # Create a list of the angles at which to rotate the arms.
    angles = [(360 / n_arms) * i for i in range(n_arms)]

    for i, angle in enumerate(angles):
        points = _generate_spiral(n_samples, start_angle, stop_angle, angle, noise_std, random_seed=random_seed)
        labels = np.full((n_samples,), i)
        x = np.concatenate((x, points))
        y = np.concatenate((y, labels))

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(x, y, random_seed)

    (x_train, x_test, x_valid) = _normalise_by_train(x_train, x_test, x_valid)

    return _zip_dataset(x_train, y_train), _zip_dataset(x_test, y_test), _zip_dataset(x_valid, y_valid)


def _zip_dataset(x, y):
    return list(zip(x, y))


def _normalise_by_train(train_data, test_data, valid_data):
    train_mean, train_std = train_data.mean(axis=0), train_data.std(axis=0)

    train_data_norm = ((train_data - train_mean) / train_std).astype(np.float32)
    valid_data_norm = ((valid_data - train_mean) / train_std).astype(np.float32)
    test_data_norm = ((test_data - train_mean) / train_std).astype(np.float32)
    return (train_data_norm, test_data_norm, valid_data_norm)


def _train_valid_test_split(x, y, seed):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=seed)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)

    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
