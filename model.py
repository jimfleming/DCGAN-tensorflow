import tensorflow as tf

from ops import BatchNorm, binary_cross_entropy_with_logits, conv2d, deconv2d, linear

class DCGAN(object):
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = 64
        self.image_shape = [64, 64, 3]

        self.z_dim = 100

        self.gf_dim = 64
        self.df_dim = 64

        self.gfc_dim = 1024
        self.dfc_dim = 1024

        self.c_dim = 3

        self.d_bn1 = BatchNorm(self.batch_size, name='d_bn1')
        self.d_bn2 = BatchNorm(self.batch_size, name='d_bn2')
        self.d_bn3 = BatchNorm(self.batch_size, name='d_bn3')

        self.g_bn0 = BatchNorm(self.batch_size, name='g_bn0')
        self.g_bn1 = BatchNorm(self.batch_size, name='g_bn1')
        self.g_bn2 = BatchNorm(self.batch_size, name='g_bn2')
        self.g_bn3 = BatchNorm(self.batch_size, name='g_bn3')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + self.image_shape, name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z)
        self.D = self.discriminator(self.images)

        self.sampler = self.sampler(self.z)
        self.D_ = self.discriminator(self.G, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        lr = 2e-4
        beta1 = 0.5
        self.d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.g_loss, var_list=g_vars)

    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = tf.nn.relu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = tf.nn.relu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
        h2 = tf.nn.relu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
        h3 = tf.nn.relu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4)

    def generator(self, z, y=None):
        # project `z` and reshape
        z_, self.h0_w = linear(z, self.gf_dim * 8*4 * 4, 'g_h0_lin', with_w=True)
        h0 = tf.reshape(z_, [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1, self.h1_w = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim * 4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2, self.h2_w = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim * 2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim * 1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        # project `z` and reshape
        h0 = tf.reshape(linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin'), [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)
