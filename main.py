import os
from glob import glob

import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import get_image, save_images

def main():
    with tf.Session() as sess:
        num_epoch = 25

        train_size = np.inf
        batch_size = 64
        image_size = 64

        checkpoint_interval = 1

        model = DCGAN(sess, batch_size=batch_size)
        data = glob(os.path.join("./data", "celebA", "*.jpg"))

        sess.run(tf.initialize_all_variables())

        sample_z = np.random.uniform(-1.0, 1.0, size=(model.sample_size , model.z_dim))
        sample_files = data[0:model.sample_size]

        sample = [get_image(sample_file, image_size, is_crop=True) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        for epoch in range(num_epoch):
            data = glob(os.path.join("./data", 'celebA', "*.jpg"))
            batch_idxs = min(len(data), train_size) / batch_size

            for idx in range(0, batch_idxs):
                batch_files = data[idx*batch_size:(idx+1)*batch_size]
                batch = [get_image(batch_file, image_size, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim]).astype(np.float32)

                # update d network
                sess.run(model.d_optim, feed_dict={
                    model.images: batch_images,
                    model.z: batch_z
                })

                # update g network
                sess.run(model.g_optim, feed_dict={
                    model.z: batch_z
                })

                # run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                sess.run(model.g_optim, feed_dict={
                    model.z: batch_z
                })

                batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim]).astype(np.float32)
                samples, d_loss, g_loss = sess.run([model.sampler, model.d_loss, model.g_loss], feed_dict={
                    model.z: sample_z,
                    model.images: sample_images
                })

                save_images(samples, [8, 8], './samples/train_{0}_{1}.png'.format(epoch, idx))
                print('loss: {0} (D) {1} (G)'.format(d_loss, g_loss))

if __name__ == '__main__':
    main()
