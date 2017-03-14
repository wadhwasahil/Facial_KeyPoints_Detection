import data_helpers
import tensorflow as tf
import numpy as np

num_keypoints = 30
graph = tf.Graph()
model_variable_scope = "1fc_b36_e1000"
image_size = 96
learning_rate = 0.01
momentum = 0.9


def fully_connected(input, size):
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.0)
                             )
    return tf.matmul(input, weights) + biases


def model_pass(input):
    with tf.variable_scope('hidden'):
        hidden = fully_connected(input, size=100)
    relu_hidden = tf.nn.relu(hidden)
    with tf.variable_scope('out'):
        prediction = fully_connected(relu_hidden, size=num_keypoints)
    return prediction


with graph.as_default():
    tf_x_batch = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    tf_y_batch = tf.placeholder(tf.float32, shape=(None, num_keypoints))

    # Training computation.
    with tf.variable_scope(model_variable_scope):
        predictions = model_pass(tf_x_batch)

    loss = tf.reduce_mean(tf.square(predictions - tf_y_batch))

    # Optimizer.
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        use_nesterov=True
    ).minimize(loss)

    with tf.Session(graph=graph) as session:
        # Initialise all variables in the graph
        session.run(tf.global_variables_initializer())
        X, y = data_helpers.get_data()
        batches = data_helpers.batch_iter(zip(X, y), batch_size=36, num_epochs=1001, shuffle=True)
        for batch in batches:
            X_train, y_train = zip(*batch)
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            _, Loss = session.run([optimizer, loss], feed_dict={
                tf_x_batch: X_train,
                tf_y_batch: y_train
            }
                        )
            print(Loss)
            print("===========================================")
