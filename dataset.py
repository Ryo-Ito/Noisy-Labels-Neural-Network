import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata


def load_noisy_mnist(noise_ratio=0.):
    """
    load MNIST with noisy labels

    Parameters
    ----------
    noise_ratio : float
        ratio of noisy labels in training

    Returns
    -------
    x_train : (60000, 784) np.ndarray
        flattened training images
    x_test : (10000, 784) np.ndarray
        flattened test images
    y_train : (60000,) np.ndarray
        training labels
    y_test : (10000,) np.ndarray
        test labels
    """
    mnist = fetch_mldata("MNIST original")
    x = np.float32(mnist.data)
    x /= np.max(x, axis=1, keepdims=True)
    y = mnist.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)
    # indices = np.random.choice(60000, int(60000 * noise_ratio), False)
    indices = np.arange(int(60000 * noise_ratio))
    y_train[indices] = np.random.randint(0, 10, len(indices))
    y_train = np.int32(y_train)
    y_test = np.int32(y_test)
    return x_train, x_test, y_train, y_test
