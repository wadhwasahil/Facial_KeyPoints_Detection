import numpy as np
import tensorflow as tf
import datetime
import data_helpers
import linear_nn


def linear_NN(X, y):
    graph = tf.Graph()
    with graph.as_default():
        nn = linear_nn.nn_linear(X, y)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001,
            momentum=0.9,
            use_nesterov=True
        ).minimize(nn.loss)
        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            batches = data_helpers.batch_iter(zip(X, y), batch_size=64, num_epochs=10, shuffle=True)
            for batch in batches:
                X_train, y_train = zip(*batch)
                feed_dict = {nn.input_x: np.asarray(X_train), nn.input_y: np.asarray(y_train)}
                _, step, loss, predictions = session.run([optimizer, global_step, nn.loss, nn.predictions], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))

X, y = data_helpers.get_data()
linear_NN(X, y)