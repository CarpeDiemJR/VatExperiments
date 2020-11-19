import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


def get_data(image_class=(4, 9), imbalance_rate=1.0, summary=False, shadow=None):
    var = locals()
    tf.keras.backend.set_floatx('float32')

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # extract by labels
    for i in image_class:
        var["x_train_" + str(i)] = x_train[y_train == i]
        var["y_train_" + str(i)] = y_train[y_train == i]

    # sample by imbalance_rate
    sample_number = int(imbalance_rate * var["x_train_" + str(image_class[-1])].shape[0])
    var["x_train_" + str(image_class[-1])] = \
        tf.random.shuffle(var["x_train_" + str(image_class[-1])])[:sample_number]
    var["y_train_" + str(image_class[-1])] = var["y_train_" + str(image_class[-1])][:sample_number]

    imbalance_rate = sample_number / var["x_train_" + str(image_class[0])].shape[0]

    if summary:
        print("Sampled ...")
        for i in image_class:
            print("    %4d images of digit %d" % (var["x_train_" + str(i)].shape[0], i))
        print("imbalanced_rate: ", imbalance_rate)

    images = np.concatenate([var["x_train_" + str(i)] for i in image_class], axis=0)
    labels = np.concatenate([var["y_train_" + str(i)] for i in image_class], axis=0)

    images = images / 255
    images = images.astype("float32")

    if shadow:
        labels = shadow_label(labels, ratio=(1-shadow, shadow))

    assert images.shape[0] == labels.shape[0]

    return images.reshape((-1, 28, 28, 1)), labels.reshape((-1, 1))


def shadow_label(labels, ratio=(1.0, 0.0)):
    shadowed = np.random.choice([False, True], size=labels.shape, p=ratio)
    labels = labels.astype('float')
    labels[shadowed] = np.nan
    return labels.reshape(-1, 1)


def shuffle(images, labels):
    index = [i for i in range(images.shape[0])]
    random.shuffle(index)
    images = images[index]
    labels = labels[index]
    return images, labels


def make_batch(images, labels, batch_size=32):
    set_size = labels.shape[0]
    images, labels = shuffle(images, labels)
    batches = [(images[stop-batch_size: stop], labels[stop-batch_size: stop])
               for stop in range(batch_size, set_size, batch_size)]
    return batches


def get_test(image_class=(4, 9)):
    var = locals()

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # extract by labels
    for i in image_class:
        var["x_train_" + str(i)] = x_train[y_train == i]
        var["y_train_" + str(i)] = y_train[y_train == i]

    images = np.concatenate([var["x_train_" + str(i)] for i in image_class], axis=0)
    labels = np.concatenate([var["y_train_" + str(i)] for i in image_class], axis=0)

    images = images / 255
    images = images.astype("float32")

    assert images.shape[0] == labels.shape[0]

    return images.reshape((-1, 28, 28, 1)), labels.reshape((-1, 1))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    print("training features shape:", train_images.shape)
    print("training labels shape:", train_labels.shape)
    for i in range(10):
        print("     # of training images of digit {}:".format(i),
              train_images[train_labels == i].shape[0])
    print("testing features shape:", test_images.shape)

    images, labels = get_data(imbalance_rate=0.2, summary=True)
    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    images, labels = shuffle(images, labels)

    print("Shawdowed shape:", shadow_label(labels.copy()).shape)
    print("Number of batches:", len(make_batch(images, labels)))

    fig, axs = plt.subplots(4, 5, figsize=(8, 10))
    for ax in axs.reshape(-1):
        index = np.random.choice(images.shape[0], 1)
        ax.imshow(images[index].reshape((28, 28, 1)), cmap="gray")
        ax.set_title("digit %d" % labels[index])
    fig.suptitle('Random sample and display')
    plt.show()
