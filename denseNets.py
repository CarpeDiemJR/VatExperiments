import tensorflow as tf
from tensorflow.keras import layers


tf.keras.backend.set_floatx('float32')


def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.math.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def get_model(input_shape, name="mlp", num_layers=3, summary=False):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Flatten(input_shape=input_shape))

    for _ in range(num_layers):
        model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid', name="prob_output"))

    if summary:
        print(model.summary())

    return model


if __name__ == "__main__":

    input_shape = (28, 28, 1)

    test = tf.random.truncated_normal(shape=(1,)+input_shape, stddev=0.1)

    model = get_model(input_shape=input_shape, summary=True)
    print("Test:", test.shape)
    print("Test Output:", model(test))