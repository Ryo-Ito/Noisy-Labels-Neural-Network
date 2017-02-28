import argparse
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from dataset import load_noisy_mnist
from model import MLP


parser = argparse.ArgumentParser(
    description="training noisy labels neural-network")
parser.add_argument(
    "--batch_size", "-b", default=50, type=int,
    help="number of samples in each minibatch")
parser.add_argument(
    "--epoch", "-e", default=5, type=int,
    help="number of sweeps over the dataset to train")
parser.add_argument(
    "--em_step", "-s", default=5, type=int,
    help="number of EM steps")
parser.add_argument(
    "--gpu", "-g", default=-1, type=int,
    help="negative value indicates no gpu, default=-1")
parser.add_argument(
    "--noise_ratio", "-r", default=0.1, type=float,
    help="ratio of noisy labels, default=0.1")
args = parser.parse_args()
print(args)


def maximize_network(x_train, x_test, y_train, y_test):
    model = MLP()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
        x_train = xp.asarray(x_train)
        x_test = xp.asarray(x_test)
        y_train = xp.asarray(y_train)
        y_test = xp.asarray(y_test)
    else:
        xp = np

    optimizer = chainer.optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    for i in range(1, args.epoch + 1):
        for j in range(0, len(x_train), args.batch_size):
            model.cleargrads()
            logit = model(x_train[j: j + args.batch_size])
            loss = -F.sum(F.log(F.softmax(logit)) * y_train[j: j + args.batch_size])
            loss.backward()
            optimizer.update()
        accuracy = F.accuracy(model(x_test), y_test)
        print("epoch {0:02d}, accuracy {1}".format(i, accuracy.data))
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

    return model


def maximize_matrix(y_train, z_train):
    m = np.zeros((10, 10))
    for j in range(10):
        indices = np.where(z_train == j)
        for i in range(10):
            m[j, i] = np.sum(y_train[indices, i]) / np.sum(y_train[:, i])
    return m


def expectation(model, x_train, z_train, noise_matrix):
    y_train = np.zeros((len(x_train), 10))
    for i in range(0, len(x_train), args.batch_size):
        proba = F.softmax(model(x_train[i: i + args.batch_size])).data
        proba *= noise_matrix[z_train[i: i + args.batch_size]]
        proba /= np.sum(proba, axis=1, keepdims=True)
        y_train[i: i + args.batch_size] = proba
    return y_train


def main():
    x_train, x_test, z_train, y_test = load_noisy_mnist(args.noise_ratio)
    y_train = np.copy(z_train)
    y_train = np.int32(LabelBinarizer().fit_transform(y_train))
    noise_matrix = np.ones((10, 10)) / 10.

    model = maximize_network(x_train, x_test, y_train, y_test)

    for i in range(1, args.em_step + 1):
        print("===============================")
        print("EM step {0:02d}".format(i))
        y_train = expectation(model, x_train, z_train, noise_matrix)
        noise_matrix = maximize_matrix(y_train, z_train)
        model = maximize_network(x_train, x_test, y_train, y_test)

    y_train = expectation(model, x_train, z_train, noise_matrix)
    y_train = np.argmax(y_train, -1)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
        plt.title("{0}->{1}".format(z_train[i], y_train[i]), fontsize=8, y=.9)
        plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
