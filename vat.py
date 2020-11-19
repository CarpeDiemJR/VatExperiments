"""
Here the most difficult part is to find a good r_qadv.
References:
    1. https://github.com/takerum/vat_tf/blob/master/vat.py
    2. ALgorithm 1, http://arxiv.org/abs/1704.03976
"""

import tensorflow as tf
from VatExperiments.generateData import get_data
from VatExperiments.denseNets import get_model, kl_divergence_with_logit


input_shape = (28, 28, 1)

model = get_model(input_shape=input_shape, name="MLP")


def get_normalized_vector(d):
    d /= (1e-12 + tf.math.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keepdims=True))
    d /= tf.sqrt(1e-6 + tf.math.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keepdims=True))
    return d


def generative_virtual_adversarial_perturbation(x, model, d=None, num_power_iterations=2, xi=1e-6, epsilon=8.0):
    '''
    :param x: images
    :param model: model, mlp/cnn/...
    :param d: perturbation
    :param num_power_iterations: for 1 image, times you want to train with d
    :param xi: see algorithm 1
    :return: a good d
    '''
    if not d:
        d = tf.random.normal(shape=tf.shape(x)[1:], dtype=tf.float32)

    for _ in range(num_power_iterations):
        d = xi * get_normalized_vector(d)
        with tf.GradientTape() as tape:
            tape.watch(d)
            logit_p = model(x)
            logit_m = model(x + d)
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tape.gradient(dist, [d])[0]

        d = tf.stop_gradient(grad)

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, model, name="vat_loss"):
    r_vadv = generative_virtual_adversarial_perturbation(x, model)

    logit_p = model(x)
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


@tf.function
def custom_loss(y_true, y_pred, logit, logit_r, alpha=0.5):
    loss = tf.reduce_sum(tf.losses.binary_crossentropy(y_true, y_pred))
    loss += alpha * tf.reduce_sum(tf.losses.kl_divergence(tf.nn.softmax(logit),
                                                   tf.nn.softmax(logit_r)))
    return loss


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float32')
    images, labels = get_data()
    model = get_model((28, 28, 1))

    print("Start Testing ...")
    print("-------------Test for KL func-------------")
    d = tf.random.normal(shape=(1, 28, 28, 1), dtype=tf.float32)
    d = 8 * get_normalized_vector(d)
    logit_p = model(images[:1])
    logit_m = model(images[:1] + d)
    dist = kl_divergence_with_logit(logit_p, logit_m)
    print("KL({}, {}) = {}.".format(logit_p.numpy(), logit_m.numpy(), dist.numpy()))
    print("-------------Test for VAT-----------------")
    for i in range(8):
        vat_loss = virtual_adversarial_loss(x=images[i].reshape(-1, 28, 28, 1), model=model)
        print("VAT loss = ", vat_loss.numpy())
    r_d = generative_virtual_adversarial_perturbation(x=images[:32].reshape(-1, 28, 28, 1), model=model)
    print("d shape：", r_d.numpy().shape)

    print("-------------Test finished----------------")