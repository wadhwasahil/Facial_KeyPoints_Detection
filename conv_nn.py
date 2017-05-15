import tensorflow as tf


class conv_nn(object):
    def __init__(self, num_classes, img_dim=96):
        self.x = tf.placeholder(tf.float32, [None, img_dim * img_dim])
        self.y = tf.placeholder(tf.float32, [None, num_classes])

        self.x_expanded = tf.expand_dims(tf.reshape(self.x, [-1, img_dim, img_dim]), axis=-1)

        with tf.variable_scope("conv1"):
            W = tf.get_variable('weights', shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(self.x_expanded, W, strides=[1, 1, 1, 1], padding='VALID')
            relu = tf.nn.relu(conv + B)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("conv2"):
            W = tf.get_variable('weights', shape=[2, 2, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='VALID')
            relu = tf.nn.relu(conv + B)
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        shape = pool.get_shape().as_list()
        x_2 = shape[1] * shape[2] * shape[3]
        self.flatten = tf.reshape(pool, [-1, shape[1] * shape[2] * shape[3]])

        with tf.variable_scope("fc1"):
            W = tf.get_variable('weights', shape=[x_2, 100], initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('biases', shape=[100], initializer=tf.constant_initializer(0.0))
            h = tf.matmul(self.flatten, W) + B

        with tf.variable_scope("fc2"):
            W = tf.get_variable('weights', shape=[100, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('biases', shape=[num_classes], initializer=tf.constant_initializer(0.0))
            h = tf.nn.relu(tf.matmul(h, W) + B)

        self.scores = h
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(self.scores - self.y))
