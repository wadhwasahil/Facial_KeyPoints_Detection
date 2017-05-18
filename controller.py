import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pandas as pd
import conv_nn
import data_helpers
import linear_nn

num_epochs = 10


def get_index(str):
    if str == "left_eye_center_x":
        return 0
    if str == "left_eye_center_y":
        return 1
    if str == "right_eye_center_x":
        return 2
    if str == "right_eye_center_y":
        return 3
    if str == "left_eye_inner_corner_x":
        return 4
    if str == "left_eye_inner_corner_y":
        return 5
    if str == "left_eye_outer_corner_x":
        return 6
    if str == "left_eye_outer_corner_y":
        return 7
    if str == "right_eye_inner_corner_x":
        return 8
    if str == "right_eye_inner_corner_y":
        return 9
    if str == "right_eye_outer_corner_x":
        return 10
    if str == "right_eye_outer_corner_y":
        return 11
    if str == "left_eyebrow_inner_end_x":
        return 12
    if str == "left_eyebrow_inner_end_y":
        return 13
    if str == "left_eyebrow_outer_end_x":
        return 14
    if str == "left_eyebrow_outer_end_y":
        return 15
    if str == "right_eyebrow_inner_end_x":
        return 16
    if str == "right_eyebrow_inner_end_y":
        return 17
    if str == "right_eyebrow_outer_end_x":
        return 18
    if str == "right_eyebrow_outer_end_y":
        return 19
    if str == "nose_tip_x":
        return 20
    if str == "nose_tip_y":
        return 21
    if str == "mouth_left_corner_x":
        return 22
    if str == "mouth_left_corner_y":
        return 23
    if str == "mouth_right_corner_x":
        return 24
    if str == "mouth_right_corner_y":
        return 25
    if str == "mouth_center_top_lip_x":
        return 26
    if str == "mouth_center_top_lip_y":
        return 27
    if str == "mouth_center_bottom_lip_x":
        return 28
    if str == "mouth_center_bottom_lip_y":
        return 29


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


def conv_NN(X, y, test=False):
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            if test == True:
                checkpoint_dir = "data/1495080366/checkpoints"
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(session, checkpoint_file)
                input_x = graph.get_operation_by_name("x_input").outputs[0]
                scores = graph.get_operation_by_name("scores").outputs[0]
                with open("IdLookupTable.csv") as f:
                    lines = [line.split() for line in f]
                locations = []
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    img_id = int(line[0].split(',')[1])
                    idx = int(get_index(line[0].split(',')[2][1:-1]))
                    X_train = np.asarray(X[img_id - 1])
                    feed_dict = {input_x: X_train.reshape(1, len(X_train))}
                    score = session.run([scores], feed_dict)
                    location = ((score[0] * 48) + 48)[0][idx]
                    if location > 96.0:
                        location = 96
                    locations.append(location)
                df = pd.DataFrame(locations, columns=["Location"])
                df.to_csv("submit.csv", index=False)


            else:
                cnn = conv_nn.conv_nn(num_classes=30)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=0.001,
                    momentum=0.9,
                    use_nesterov=True,
                ).minimize(cnn.loss, global_step=global_step)
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "data", timestamp))
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.all_variables())
                train_loss_history = []
                session.run(tf.global_variables_initializer())
                batches = data_helpers.batch_iter(zip(X, y), batch_size=64, num_epochs=num_epochs, shuffle=True)
                for batch in batches:
                    X_train, y_train = zip(*batch)
                    feed_dict = {cnn.x: np.asarray(X_train), cnn.y: np.asarray(y_train)}
                    _, step, loss, scores = session.run([optimizer, global_step, cnn.loss, cnn.scores], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                    if step % 10 == 0 and test == False:
                        path = saver.save(session, checkpoint_prefix, global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
                    train_loss_history.append(loss)
                if test == False:
                    x_axis = np.arange(step)
                    plt.plot(x_axis, train_loss_history, "b-", linewidth=2, label="train")
                    plt.grid()
                    plt.legend()
                    plt.ylabel("loss")
                    plt.show()


'''Training part'''
# X, y = data_helpers.get_data()
# conv_NN(X, y)

'''Testing part'''
X, y = data_helpers.get_data(test=True)
conv_NN(X, y, True)
