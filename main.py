from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import pickle
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import train_test_split

from utils import save_images
from model import DCGAN
from dataset import DataIterator

def main():
    batch_size = 128

    with open('dataset/data_10_tf.pkl', 'rb') as f:
        X_train_raw, y_train_raw, _, _, _ = pickle.load(f)

    X_train_raw = X_train_raw.astype(np.float32)

    mean_train = X_train_raw.mean()
    std_train = X_train_raw.std()

    X_train_raw = X_train_raw - mean_train
    X_train_raw = X_train_raw / std_train

    print('train', 'mean', X_train_raw.mean(), 'std', X_train_raw.std())

    with tf.Session() as sess:
        num_epoch = 20
        checkpoint_interval = 10

        model = DCGAN(sess, batch_size=batch_size)
        summary_writer = tf.train.SummaryWriter('logs_{0}/'.format(int(time.time())), sess.graph_def)
        sess.run(tf.initialize_all_variables())

        X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw, \
            test_size=model.sample_size, random_state=41)

        dataset_iter = DataIterator(X_train, y_train, batch_size)

        sample_images = X_valid
        sample_z = np.random.uniform(-1.0, 1.0, size=(model.sample_size, model.z_dim))

        d_overpowered = False

        step = 0
        for epoch in range(num_epoch):
            for batch_images, _ in dataset_iter.iterate():
                batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim])

                # update d network
                if not d_overpowered:
                    sess.run(model.d_optim, feed_dict={ model.x: batch_images, model.z: batch_z })

                # update g network
                sess.run(model.g_optim, feed_dict={ model.z: batch_z })

                if step % checkpoint_interval == 0:
                    d_loss, g_loss, summary = sess.run([
                        model.d_loss,
                        model.g_loss,
                        model.merged
                    ], feed_dict={
                        model.x: sample_images,
                        model.z: sample_z
                    })
                    summary_writer.add_summary(summary, step)

                    if epoch == 0:
                        d_overpowered = d_loss < g_loss / 2
                    else:
                        d_overpowered = d_loss < g_loss

                    samples = sess.run(model.G, feed_dict={
                        model.x: sample_images,
                        model.z: sample_z
                    })

                    samples *= std_train
                    samples += mean_train

                    samples_path = './samples/train_{0}_{1}.png'.format(epoch, step)
                    save_images(samples, [16, 8], samples_path)
                    print('[{0}, {1}] loss: {2} (D) {3} (G) (d overpowered?: {4})'.format(epoch, step, d_loss, g_loss, d_overpowered))

                step += 1

if __name__ == '__main__':
    main()
