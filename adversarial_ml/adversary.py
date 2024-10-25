import foolbox as fb
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from foolbox.attacks import L0FMNAttack, L1FMNAttack, L2FMNAttack, LInfFMNAttack

import preprocessing


def plot_all_samples(advs):
    num_classes = 10

    # Initialize a grid to hold images for each (true_label, predicted_label) pair
    grid = [[None for _ in range(num_classes)] for _ in range(num_classes)]

    # Fill the grid with adversarial images based on true and predicted labels
    for i in range(10):
        for j in range(10):
            grid[i][j] = advs[i][j].numpy().squeeze()

    fig, axes = plt.subplots(num_classes, num_classes, figsize=(10, 10))

    for i in range(num_classes):
        for j in range(num_classes):
            if grid[i][j] is not None:
                axes[i, j].imshow(grid[i][j], cmap='gray')
            axes[i, j].axis('off')

    for ax, col in zip(axes[0], range(num_classes)):
        ax.set_title(col)

    for num, ax in zip(range(10), axes[:, 0]):
        ax.set_ylabel(num, rotation=0, size='large', labelpad=10)
        ax.axis('on')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.text(0.5, 0.04, 'Input class', ha='center', va='center')
    fig.text(0.06, 0.5, 'Output classification', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def perform_attack(fb_model, target):
    x_train, y_train_enc, x_test, y_test_enc = preprocessing.get_dataset()
    y_train = np.argmax(y_train_enc, axis=1)

    _, unique_idx = np.unique(y_train, return_index=True)
    x_train_tf = tf.convert_to_tensor(x_train[unique_idx], dtype=tf.float32)

    attack = L1FMNAttack()
    criterion = fb.criteria.TargetedMisclassification(np.full((10,), target))
    advs, _, is_adv = attack(fb_model, x_train_tf, criterion, epsilons=None)

    num_successful = tf.reduce_sum(tf.cast(is_adv, tf.int32)).numpy()
    print(f"Number of successful adversarial examples: {num_successful}")

    misclassified_labels = fb_model(advs).numpy().argmax(axis=1)
    print(f"Misclassified labels: {misclassified_labels}")

    return advs


if __name__ == '__main__':
    model = load_model('mnist_cnn.keras')
    print("Loaded model")
    fb_model = fb.TensorFlowModel(model, (0.0, 1.0))

    advs = []
    for i in range(10):
        advs.append(perform_attack(fb_model, i))
    plot_all_samples(advs)