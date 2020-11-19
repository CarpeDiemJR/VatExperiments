"""
References:
    1. https://github.com/takerum/vat_tf/blob/master/vat.py
"""

import tensorflow as tf
from VatExperiments.generateData import get_data
from VatExperiments.denseNets import get_model, kl_divergence_with_logit


input_shape = (28, 28, 1)

model = get_model(input_shape=input_shape, name="MLP")


def logit(x):
    return model(x)


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d


def generative_virtual_adversarial_perturbation(x, logit, num_power_iterations=1, xi=1e-6, epsilon=8.0):
    d = tf.random.normal(shape=tf.shape(x))

    for _ in range(num_power_iterations):
        d = xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, name="vat_loss"):
    r_vadv = generative_virtual_adversarial_perturbation(x, logit)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


def generate_adversarial_perturbation(x, loss, epsilon=8.0):
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return epsilon * get_normalized_vector(grad)


def ce_loss(logit, y):
    return tf.nn.softmax_cross_entropy_with_logits(y, logit


def adversarial_loss(x, y, loss, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = model(x + r_adv, update_batch_stats=False)
    loss = ce_loss(logit, y)
    return tf.identity(loss, name=name)


if __name__ == "__main__":
    print(0)