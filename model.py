import numpy as np
import tensorflow as tf

from ops import BatchNorm, conv2d, deconv2d, linear, binary_cross_entropy_with_logits, lrelu

class DCGAN(object):
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.image_shape = [48, 64, 3]

        self.z_dim = 100
        self.c_dim = 10

        self.gf_dim = 128
        self.df_dim = 64

        self.d_bn0 = BatchNorm(self.batch_size, name='d_bn0')
        self.d_bn1 = BatchNorm(self.batch_size, name='d_bn1')
        self.d_bn2 = BatchNorm(self.batch_size, name='d_bn2')
        self.d_bn3 = BatchNorm(self.batch_size, name='d_bn3')
        self.d_bn4 = BatchNorm(self.batch_size, name='d_bn4')

        self.g_bn0 = BatchNorm(self.batch_size, name='g_bn0')
        self.g_bn1 = BatchNorm(self.batch_size, name='g_bn1')
        self.g_bn2 = BatchNorm(self.batch_size, name='g_bn2')
        self.g_bn3 = BatchNorm(self.batch_size, name='g_bn3')

        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='x')
        self.c = tf.placeholder(tf.float32, [self.batch_size, self.c_dim], name='c')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.G = self.generator(self.z, self.c)
        self.D = self.discriminator(self.x, self.c)
        self.D_ = self.discriminator(self.G, self.c, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        tf.histogram_summary("D", self.D)
        tf.histogram_summary("D_", self.D_)

        tf.scalar_summary("d_loss_real", self.d_loss_real)
        tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        tf.scalar_summary("d_loss", self.d_loss)
        tf.scalar_summary("g_loss", self.g_loss)

        self.merged = tf.merge_all_summaries()

        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        lr = 2e-4
        beta1 = 0.5
        self.d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.g_loss, var_list=g_vars)

    def discriminator(self, x, c, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = linear(c, self.image_shape[0] * self.image_shape[1], name='d_h0_lin')
        h0 = tf.reshape(h0, [-1] + self.image_shape[:2] + [1])
        h0 = self.d_bn0(tf.nn.elu(h0))
        h0 = tf.concat(3, [x, h0], name='concat')

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 2, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 4, name='d_h3_conv')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 8, name='d_h4_conv')))
        h5 = tf.nn.sigmoid(linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h5_lin'))
        return h5

    def generator(self, z, c):
        zc = tf.concat(1, [z, c])

        h0 = linear(zc, self.gf_dim * 8 * 3 * 4, name='g_h0_lin')
        h0 = tf.reshape(h0, [-1, 3, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, [self.batch_size, 6, 8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, [self.batch_size, 12, 16, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, [self.batch_size, 24, 32, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, [self.batch_size, 48, 64, 3], name='g_h4')
        h4 = tf.nn.tanh(h4)
        return h4
