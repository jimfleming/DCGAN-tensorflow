import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

class BatchNorm(object):
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm", collections=None):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size
            self.collections = collections

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]], \
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), collections=self.collections)
            self.beta = tf.get_variable("beta", [shape[-1]], \
                initializer=tf.constant_initializer(0.0), collections=self.collections)

            mean, variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(x, mean, variance, \
                self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def binary_cross_entropy_with_logits(logits, targets, name=None):
    epsilon = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + epsilon) + (1. - logits) * tf.log(1. - targets + epsilon)))

def conv2d(x, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, name="conv2d", collections=None):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [k_h, k_w, x.get_shape()[-1], output_dim], \
            initializer=tf.truncated_normal_initializer(stddev=0.02), \
            collections=collections)
        b = tf.get_variable('b', [output_dim], \
            initializer=tf.constant_initializer(0.1), \
            collections=collections)

        conv = tf.nn.conv2d(x, W, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def deconv2d(x, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, name="deconv2d", collections=None):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [k_h, k_h, output_shape[-1], x.get_shape()[-1]], \
            initializer=tf.truncated_normal_initializer(stddev=0.02), \
            collections=collections)
        b = tf.get_variable('b', [output_shape[-1]], \
            initializer=tf.constant_initializer(0.1), \
            collections=collections)

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv

def linear(x, output_shape, name=None, collections=None):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        W = tf.get_variable("W", [shape[1], output_shape], \
            initializer=tf.truncated_normal_initializer(stddev=0.02), \
            collections=collections)
        b = tf.get_variable("bias", [output_shape], \
            initializer=tf.constant_initializer(0.1), \
            collections=collections)
        return tf.matmul(x, W) + b
