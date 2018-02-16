import random

import tensorflow as tf
import numpy as np
import pickle
import scipy.io as sio
import sys


class LinearRegression:
    def __init__(self, training_epochs=500, learning_rate=0.05):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs  # this maybe too much?
        self.n_dim = 2
        self.n_output = 1
        self.batch_size = 10
        # self.W = tf.Variable(np.zeros((self.n_dim, self.n_output)), name="weight")
        # self.b = tf.Variable(np.zeros((1, self.n_output)), name="bias")

    def delete_para(self):
        # del self.learning_rate
        # del self.training_epochs
        self.W = None
        self.b = None
        self.training_epochs = None
        self.training_epochs = None
        self.n_dim = None
        self.n_output = None
        self.batch_size = None
        self.X = None
        self.Y = None
        self.pred = None
        self.init = None
        self.cost = None
        self.optimizer = None
        # del self.n_dim
        # del self.n_output
        # del self.batch_size

        return
        # print 'Destructors'

    def read_weights(self, weights=None, bias=None):
        if weights is not None:
            self.W = self.weight_initialization(False, weights)
        else:
            self.W = self.weight_initialization()
        if bias is not None:
            self.b = self.bias_initialization(False, bias)
        else:
            self.b = self.bias_initialization()

    def weight_initialization(self, initial_flag=True, values=None):
        if initial_flag == True:
            initial_value = np.random.randn(self.n_dim, self.n_output)
        else:
            initial_value = values

        weight = tf.Variable(initial_value, name="weight")
        return weight

    def bias_initialization(self, initial_flag=True, values=None):
        if initial_flag == True:
            initial_value = np.random.randn(1, self.n_output)
        else:
            initial_value = values
        bias = tf.Variable(initial_value, name="bias")
        return bias

    def linear_regression_model(self):
        # tf Graph Input
        self.X = tf.placeholder("float64", [None, self.n_dim])  # CurrentObs
        self.Y = tf.placeholder("float64", [None, self.n_output])  # Q_home and Q_away

        # Construct a linear model
        self.pred = tf.add(tf.matmul(self.X, self.W), self.b)

        # Mean squared error
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.pred - self.Y)))

        # Gradient descent
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def readout_linear_regression_model(self):
        # tf Graph Input
        self.X = tf.placeholder("float64", [None, self.n_dim])  # CurrentObs

        # Construct a linear model
        self.pred = tf.add(tf.matmul(self.X, self.W), self.b)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def compute_average_difference(self, listA, listB):
        diff_all = float(0)
        max_diff = 0
        for i in range(0, len(listA)):
            list_a = listA[i]
            list_b = listB[i]
            diff = float(0)
            for j in range(0, len(list_a)):
                sub_diff = abs(float(list_a[j]) - float(list_b[j]))
                max_diff = sub_diff if sub_diff > max_diff else max_diff
                diff += sub_diff
            diff_all += diff / len(list_a)

        return diff_all / len(listA), max_diff

    def gradient_descent(self, sess, train_X, train_Y):
        """
        Use tensorflow to do gradient descent
        :param train_X: training data (currentObs)
        :param train_Y: result value (q_values)
        :param n_samples: the number of instances
        :return: []
        """
        sess.run(self.init)

        random.sample(range(len(train_X)), len(train_X))

        # Fit all training data
        for epoch in range(self.training_epochs):

            random_number = random.sample(range(len(train_X)), len(train_X))
            train_X_reordered = [train_X[num] for num in random_number]
            train_Y_reordered = [train_Y[num] for num in random_number]

            if len(train_X) <= self.batch_size:
                cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: train_X, self.Y: train_Y})
                # print cost
            else:
                for i in range(0, len(train_X) / self.batch_size):
                    if i + 1 < (len(train_X) / self.batch_size):
                        input_, labels = train_X_reordered[
                                         i * self.batch_size:i * self.batch_size + self.batch_size], train_Y_reordered[
                                                                                                     i * self.batch_size:i * self.batch_size + self.batch_size]
                    else:
                        input_, labels = train_X_reordered[i * self.batch_size:], train_Y_reordered[
                                                                                  i * self.batch_size:]
                    cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: input_, self.Y: labels})
                    # print cost

                    # for (x, y) in zip(train_X_reordered, train_Y_reordered):
                    #     cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: [x], self.Y: [y]})
                    #     print cost

        trained_weights = sess.run(self.W)
        trained_bias = sess.run(self.b)
        # pickle.dump(temp1, open('./temp_test_save/weights_1.p', 'w'))
        # pickle.dump(temp2, open('./temp_test_save/bias_1.p', 'w'))
        #
        temp = sess.run(self.pred, feed_dict={self.X: train_X, self.Y: train_Y}).tolist()

        average_diff, max_diff = self.compute_average_difference(train_Y, temp)

        print >> sys.stderr, '(average_diff:{0}, max_diff:{1}, training_epochs:{2}, lr:{3})'.format(average_diff, max_diff, self.training_epochs, self.learning_rate)

        return trained_weights, trained_bias, average_diff


if __name__ == "__main__":
    test_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    test_y = [[1, 10, 5], [2, 11, 6]]
    with tf.Session() as sess:
        # """read weights and bias"""
        # weight = pickle.load(open('./temp_test_save/weights_1.p', 'r'))
        # bias = pickle.load(open('./temp_test_save/bias_1.p', 'r'))
        # LR = LinearRegression(weights=weight, bias=bias)
        # LR.linear_regression_model()
        # temp = LR.gradient_descent(sess=sess, train_X=test_x, train_Y=test_y)
        # print temp

        """don't read weights and bias"""
        LR = LinearRegression()
        LR.read_weights()
        LR.linear_regression_model()
        temp = LR.gradient_descent(sess=sess, train_X=test_x, train_Y=test_y)
        print temp
