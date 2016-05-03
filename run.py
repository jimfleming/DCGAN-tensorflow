from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np
import tensorflow as tf

from utils import save_images, mkdirp, md5, one_hot
from model import DCGAN
from dataset import DataIterator
from scipy.misc import imsave

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
mkdirp(CHECKPOINT_PATH)

SAMPLES_PATH = 'samples/'
mkdirp(SAMPLES_PATH)

def main():
    width, height, num_channels = (64, 48, 3)
    num_classes = 10
    batch_size = num_samples = 100
    z_dim = 100

    with tf.Session() as sess:
        sample_z = np.random.normal(size=(z_dim, num_samples // num_classes))
        sample_z = np.tile(sample_z, (num_classes, 1))
        sample_z = np.reshape(sample_z, (num_samples, -1))

        sample_labels = np.repeat(np.arange(num_classes), num_samples // num_classes)
        sample_labels = one_hot(sample_labels)

        model = DCGAN(sess, batch_size=batch_size)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model-17399')

        saver.restore(sess, checkpoint_path)

        samples = sess.run(model.G, feed_dict={
            model.c: sample_labels,
            model.z: sample_z
        })

        samples_min = samples.min()
        samples_max = samples.max()
        samples = (samples - samples_min) / (samples_max - samples_min)

        samples_path = os.path.join(SAMPLES_PATH, 'classes.png')
        save_images(samples, [num_classes, num_samples // num_classes], samples_path)
        print(samples_path, samples.shape)

if __name__ == '__main__':
    main()
