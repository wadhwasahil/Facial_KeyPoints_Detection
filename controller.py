import numpy as np
import tensorflow as tf
import datetime
import data_helpers
import linear_nn
import conv_nn
import matplotlib.pyplot as plt
num_epochs = 10

def linear_NN(X, y):
    graph = tf.Graph()
    with graph.as_default():
        nn = linear_nn.nn_linear(X, y)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001,
            momentum=0.9,
            use_nesterov=True,
        ).minimize(nn.loss, global_step=global_step)
        with tf.Session(graph=graph) as session:
            train_loss_history = []
            session.run(tf.global_variables_initializer())
            batches = data_helpers.batch_iter(zip(X, y), batch_size=64, num_epochs=num_epochs, shuffle=True)
            for batch in batches:
                X_train, y_train = zip(*batch)
                feed_dict = {nn.input_x: np.asarray(X_train), nn.input_y: np.asarray(y_train)}
                _, step, loss, predictions = session.run([optimizer, global_step, nn.loss, nn.predictions], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_loss_history.append(loss)
                # if step % 10 == 0:
                #     pass
            x_axis = np.arange(step)
            plt.plot(x_axis, train_loss_history, "b-", linewidth=2, label="train")
            plt.grid()
            plt.legend()
            plt.ylabel("loss")
            plt.show()

def conv_NN(X, y):
    graph = tf.Graph()
    with graph.as_default():
        cnn = conv_nn.conv_nn(num_classes=30)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001,
            momentum=0.9,
            use_nesterov=True,
        ).minimize(cnn.loss, global_step=global_step)
        with tf.Session(graph=graph) as session:
            train_loss_history = []
            session.run(tf.global_variables_initializer())
            batches = data_helpers.batch_iter(zip(X, y), batch_size=64, num_epochs=num_epochs, shuffle=True)
            for batch in batches:
                X_train, y_train = zip(*batch)
                feed_dict = {cnn.x: np.asarray(X_train), cnn.y: np.asarray(y_train)}
                _, step, loss = session.run([optimizer, global_step, cnn.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(loss)
                # print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_loss_history.append(loss)
            x_axis = np.arange(step)
            plt.plot(x_axis, train_loss_history, "b-", linewidth=2, label="train")
            plt.grid()
            plt.legend()
            plt.ylabel("loss")
            plt.show()


X, y = data_helpers.get_data()
# linear_NN(X, y)
conv_NN(X, y)