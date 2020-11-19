import tensorflow as tf
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float32')


@tf.function
def logsoftmax(x):
    xdev = x - tf.math.reduce_max(x, keepdims=True)
    lsm = xdev - tf.math.log(tf.reduce_sum(tf.exp(xdev), keepdims=True))
    return lsm


@tf.function
def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def get_model(input_shape, output_class=2, name="mlp", num_layers=5, summary=False):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Flatten(input_shape=input_shape))

    for _ in range(num_layers):
        model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(output_class, name="logit", dtype=tf.float32))

    if summary:
        print(model.summary())

    return model


if __name__ == "__main__":
    input_shape = (28, 28, 1)

    logit_1 = tf.random.truncated_normal(shape=(32, 1), seed=1234)
    logit_2 = tf.random.truncated_normal(shape=(32, 1), seed=1111)

    test = tf.random.truncated_normal(shape=(1,) + input_shape, stddev=0.1)

    model = get_model(input_shape=input_shape, summary=True)
    print("Test:", test.shape)
    print("Test Output:", model(test))

    from VatExperiments.generateData import get_data

    images, labels = get_data((1, 7))

    model.compile(tf.optimizers.RMSprop(1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(images, labels)
