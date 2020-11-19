import tensorflow as tf
import numpy as np
from VatExperiments.generateData import *
from VatExperiments.vat import *
from VatExperiments.denseNets import get_model

input_shape = (28, 28, 1)


num_power_iterations=2
xi=8.0
epsilon=1e-6

model = get_model(input_shape=input_shape)


def labels2binary(labels, dict):
    for key in dict.keys():
        labels[labels==int(key)] = dict[key]
    return labels


def logits2pred(logits):
    max_loc = tf.math.argmax(tf.nn.softmax(logits), axis=1)
    return tf.reshape(max_loc, (-1, 1))


def train(images, labels, dict, epochs=2, batch_size=32, num_power_iterations=2, xi=8.0,
          epsilon=1e-6, verbal=False):
    # convert the label to binary
    labels = labels2binary(labels, dict)

    # prepare training
    optimizer = tf.keras.optimizers.RMSprop(1e-4)
    counter = 0
    print("Start Training...")
    print("%d batches in %d epoch(s) to go ..." % (epochs*(images.shape[0]//batch_size), epochs))

    for epoch in range(epochs):
        print("----epoch {}-----".format(epoch))
        batches = make_batch(images, labels, batch_size=batch_size)
        for batch in batches:
            counter += 1
            if verbal:
                print(">>>>>>training iteration no.%d"% counter)
            x_train, y_train = batch
            train_step(x_train, y_train, optimizer=optimizer, num_power_iterations=num_power_iterations,
                               xi=xi, epsilon=epsilon, verbal=True)


def train_step(x, y, optimizer, num_power_iterations=num_power_iterations, xi=xi, epsilon=epsilon, verbal=False):
    # extract unlabeled
    unlabel_index = np.isnan(y).reshape(-1)
    x_labeled = x[~unlabel_index]
    y_labeled = y[~unlabel_index]
    x_unlabeled = x[unlabel_index]

    # get a good 'r'
    d = generative_virtual_adversarial_perturbation(x, model, num_power_iterations=num_power_iterations, xi=xi,
                                                    epsilon=epsilon)

    # create stats
    num_label0 = sum(y_labeled==0)
    num_label1 = sum(y_labeled==1)
    num_unlabel = x_unlabeled.shape[0]

    # gradient descent with rmsprop
    weights = model.trainable_weights
    with tf.GradientTape() as tape:
        labeled_logits = model(x_labeled)
        labeled_logits = tf.nn.softmax(labeled_logits)
        labeled_pred = labeled_logits[:, 1]
        accuracy = (sum(logits2pred(labeled_logits).numpy() == y_labeled) / x_labeled.shape[0]) * 100.0
        logit = model(x)
        logit_r = model(x + d)
        loss = custom_loss(y_labeled, labeled_pred, logit, logit_r)
        grad = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grad, weights))
    if verbal:
        print("    loss:", loss.numpy())
        print("     acc: %.2f"% accuracy)
        print(" label 0: %d" % num_label0)
        print(" label 1: %d" % num_label1)
        print("no label: %d" % num_unlabel)


if __name__ == "__main__":
    label_class = (1, 7)
    class_dict = {str(label_class[0]): 0, str(label_class[1]): 1}
    ib = 0.8
    save_dir = "References"

    images, labels = get_data(image_class=label_class, imbalance_rate=ib, shadow=.6)

    train(images, labels, class_dict, verbal=True, epochs=2)

    test_images, test_labels = get_test(label_class)
    test_logit = model(test_images)
    test_pred = logits2pred(test_logit)
    test_labels = labels2binary(test_labels, class_dict)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import os
    print(test_pred.numpy().reshape(-1))
    cm = confusion_matrix(test_labels, test_pred.numpy())
    print(cm)
    cmp = ConfusionMatrixDisplay(cm, display_labels=label_class)
    cmp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix with imbalence rate {}".format(ib))
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix_with_"+str(ib)+".pdf"))
    plt.show()

