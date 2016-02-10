import os
from glob import glob

import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import get_image, save_images
from dataset import Dataset, DataIterator

def main():
    with tf.Session() as sess:
        num_epoch = 2000

        train_size = np.inf
        batch_size = 64
        image_size = 32

        checkpoint_interval = 20

        model = DCGAN(sess, batch_size=batch_size)

        dataset = Dataset("cifar10/")
        dataset_iter = DataIterator(dataset.train_images, dataset.train_labels, batch_size)

        sess.run(tf.initialize_all_variables())

        sample_images = dataset.valid_images[:model.sample_size]
        sample_z = np.random.uniform(-1.0, 1.0, size=(model.sample_size , model.z_dim))

        for epoch in range(num_epoch):
            if epoch % checkpoint_interval == 0:
                samples, d_loss, g_loss = sess.run([model.sampler, model.d_loss, model.g_loss], feed_dict={
                    model.z: sample_z,
                    model.images: sample_images
                })

                save_images(samples, [8, 8], './samples/train_{0}.png'.format(epoch))
                print('loss: {0} (D) {1} (G)'.format(d_loss, g_loss))

            batch_images, _ = dataset_iter.next_batch()
            batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim]).astype(np.float32)

            # update d network
            sess.run(model.d_optim, feed_dict={ model.images: batch_images, model.z: batch_z })

            # update g network
            sess.run(model.g_optim, feed_dict={ model.z: batch_z })

            # # run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            # sess.run(model.g_optim, feed_dict={ model.z: batch_z })

if __name__ == '__main__':
    main()
