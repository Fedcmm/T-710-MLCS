from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def get_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train_enc = to_categorical(y_train, num_classes=10)
    y_test_enc = to_categorical(y_test, num_classes=10)

    return x_train, y_train_enc, x_test, y_test_enc