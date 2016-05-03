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
    batch_size = num_samples = 64

    with tf.Session() as sess:
        model = DCGAN(sess, batch_size=batch_size)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model-17399')

        saver.restore(sess, checkpoint_path)

        sample_z = np.random.uniform(size=(num_samples, model.z_dim))

        for i in range(num_classes):
            sample_labels = one_hot(np.ones((num_samples,), dtype=np.uint8) * i)

            samples = sess.run(model.G, feed_dict={
                model.c: sample_labels,
                model.z: sample_z
            })

            samples = (samples + 1.) / 2.

            samples_path = os.path.join(SAMPLES_PATH, 'class_{}.png'.format(i))
            save_images(samples, [8, 8], samples_path)
            print(samples_path)

if __name__ == '__main__':
    main()
