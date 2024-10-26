import os

import foolbox as fb
import numpy as np
import tensorflow as tf
import eagerpy as ep
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from foolbox.attacks import L0FMNAttack, L1FMNAttack, L2FMNAttack, LInfFMNAttack

import preprocessing

plots_dir = os.path.join(os.path.dirname(__file__), 'plots')


def plot_all_samples(advs, name):
    num_classes = 10

    # Initialize a grid to hold images for each (true_label, predicted_label) pair
    grid = [[None for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(10):
        for j in range(10):
            grid[i][j] = advs[i][j].numpy().squeeze()

    fig, axes = plt.subplots(num_classes, num_classes, figsize=(10, 10))

    for i in range(num_classes):
        for j in range(num_classes):
            axes[i, j].imshow(grid[i][j], cmap='gray')
            axes[i, j].axis('off')

    for ax, col in zip(axes[0], range(num_classes)):
        ax.set_title(col)

    for num, ax in zip(range(10), axes[:, 0]):
        ax.set_ylabel(num, rotation=0, size='large', labelpad=10)
        ax.axis('on')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.text(0.08, 0.5, 'Output classification', size='x-large', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.92, 'Input class', size='x-large', ha='center', va='center')
    fig.text(0.5, 0.97, f'Attack with {name}', size='xx-large', ha='center', va='center')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(plots_dir, f'{name}.png'))
    plt.show()


def get_fb_model():
    model = load_model('mnist_cnn.keras')
    print("Loaded model")
    return fb.TensorFlowModel(model, (0.0, 1.0))


def get_input_images():
    x_train, y_train_enc, _, _ = preprocessing.get_dataset()
    y_train = np.argmax(y_train_enc, axis=1)

    _, unique_idx = np.unique(y_train, return_index=True)
    x_train_tf = tf.convert_to_tensor(x_train[unique_idx], dtype=tf.float32)
    # y_train_tf = tf.convert_to_tensor(y_train[unique_idx])
    return x_train_tf


def perform_attack(fb_model, attack, x_train_tf, target_label):
    target = np.full((10,), target_label)
    criterion = fb.criteria.TargetedMisclassification(tf.convert_to_tensor(target))
    epsilons = [0.004, 0.01, 0.1, 1.0, 10.0, 30, 100]
    if attack.__class__ == L0FMNAttack:
        epsilons = None
    raw_adversarial, clipped_adversarial, success = attack(fb_model, x_train_tf, criterion, epsilons=epsilons)

    if epsilons is None:
        num_successful = tf.reduce_sum(tf.cast(success, tf.int32)).numpy()
        print(f"Number of successful adversarial examples: {num_successful}")
        acc = fb.accuracy(fb_model, clipped_adversarial, target)
        print(f"Accuracy: {acc * 100:4.1f} %")
        return clipped_adversarial

    success_index = -1
    for eps, advs_, succ_ in zip(epsilons, clipped_adversarial, success):
        advs_ = ep.TensorFlowTensor(advs_)
        succ_ = succ_.numpy().all()

        success_index += 1
        if eps is None:
            continue
        acc2 = fb.accuracy(fb_model, advs_, target)
        perturbation_sizes = (advs_ - x_train_tf).norms.linf(axis=(1, 2, 3)).numpy()

        print(f"  norm ≤ {eps:<6}: {acc2 * 100:4.1f} %  success: {succ_} perturbation:",
              ' '.join(map(str, perturbation_sizes)))
        if succ_:
            break

    return clipped_adversarial[success_index]


def main():
    fb_model = get_fb_model()
    x_train_tf = get_input_images()

    attacks = {
        "L0FMNAttack": L0FMNAttack(),
        "L1FMNAttack": L1FMNAttack(),
        "L2FMNAttack": L2FMNAttack(),
        "LInfFMNAttack": LInfFMNAttack()
    }
    for name, attack in attacks.items():
        print(f"\n===== Attacking with {name} =====")
        results = []
        for i in range(10):
            print(f"\nAttack with target class {i}")
            results.append(perform_attack(fb_model, attack, x_train_tf, i))
        plot_all_samples(results, name)


if __name__ == '__main__':
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    main()