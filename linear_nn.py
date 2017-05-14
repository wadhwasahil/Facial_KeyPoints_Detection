import tensorflow as tf

n_1 = 100  # no of neurons in the first layer

class nn_linear(object):
    def __init__(self, X, y):
        x_features = len(X[0])
        num_classes = len(y[0])
        self.input_x = tf.placeholder(tf.float32, [None, x_features])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])

        self.hidden_W1 = tf.Variable(tf.truncated_normal([x_features, n_1], stddev=0.01))
        self.hidden_b1 = tf.Variable(tf.constant(0.0, shape=[n_1]))
        self.result1 = tf.nn.relu(tf.matmul(self.input_x, self.hidden_W1) + self.hidden_b1)

        self.hidden_W2 = tf.Variable(tf.truncated_normal([n_1, num_classes], stddev=0.01))
        self.hidden_b2 = tf.Variable(tf.constant(0.0, shape=[num_classes]))
        self.result2 = tf.nn.relu(tf.matmul(self.result1, self.hidden_W2) + self.hidden_b2)

        with tf.variable_scope('out'):
            self.predictions = self.result2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.predictions - self.input_y))