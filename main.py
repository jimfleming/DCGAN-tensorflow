from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import train_test_split

from utils import save_images, mkdirp, md5
from model import DCGAN
from dataset import DataIterator

TESTING = os.environ.get('TESTING', False)

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_10_tf.pkl' if not TESTING else 'dataset/data_10_tf_subset.pkl')
print('md5', DATASET_PATH, md5(DATASET_PATH))

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

SAMPLES_PATH = os.path.join(SUMMARY_PATH, 'samples')
mkdirp(SAMPLES_PATH)

def main():
    batch_size = sample_size = 128

    with open(DATASET_PATH, 'rb') as f:
        X_train_raw, y_train_raw, _, _, _ = pickle.load(f)

    X_train_raw = X_train_raw.astype(np.float32)
    X_train_raw = X_train_raw / 255.

    with tf.Session() as sess:
        num_epoch = 100

        model = DCGAN(sess, batch_size=batch_size)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=5)

        now = int(time.time())
        summary_path = os.path.join(SUMMARY_PATH, 'summary_{}'.format(now))
        mkdirp(summary_path)

        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)
        sess.run(tf.initialize_all_variables())

        tf.train.write_graph(sess.graph_def, MODEL_PATH, 'model.pbtxt')

        X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw, \
            test_size=sample_size, random_state=41)

        dataset_iter = DataIterator(X_train, y_train, batch_size)

        sample_images = X_valid
        sample_labels = y_valid
        sample_z = np.random.uniform(-1.0, 1.0, size=(sample_size, model.z_dim))

        margin = 0.3
        optimize_d = False
        optimize_g = False

        step = 0
        for epoch in range(num_epoch):
            for batch_images, batch_labels in dataset_iter.iterate():
                batch_z = np.random.uniform(-1.0, 1.0, [batch_size, model.z_dim])

                if optimize_d:
                    start_time_d = time.time()
                    sess.run(model.d_optim, feed_dict={
                        model.x: batch_images,
                        model.c: batch_labels,
                        model.z: batch_z
                    })
                    end_time_d = time.time()

                if optimize_g:
                    start_time_g = time.time()
                    sess.run(model.g_optim, feed_dict={
                        model.c: batch_labels,
                        model.z: batch_z
                    })
                    end_time_g = time.time()

                d_loss, g_loss, summary = sess.run([
                    model.d_loss,
                    model.g_loss,
                    model.merged
                ], feed_dict={
                    model.x: sample_images,
                    model.c: sample_labels,
                    model.z: sample_z
                })
                summary_writer.add_summary(summary, step)

                time_d = end_time_d - start_time_d if optimize_d else 0
                time_g = end_time_g - start_time_g if optimize_g else 0

                print('[{}, {}] D ({}) {} [{}] G ({}) {} [{}]' \
                    .format(epoch, step, optimize_d, d_loss, time_d, optimize_g, g_loss, time_g))

                if d_loss < margin:
                    optimize_d = False

                if g_loss < margin:
                    optimize_g = False

                if optimize_d is False and optimize_g is False:
                    optimize_d = True
                    optimize_g = True

                step += 1

            checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model')
            saver.save(sess, checkpoint_path, global_step=step)

            samples = sess.run(model.G, feed_dict={
                model.x: sample_images,
                model.c: sample_labels,
                model.z: sample_z
            })

            samples = (samples + 1.) / 2.

            samples_path = os.path.join(SAMPLES_PATH, 'train_{}_{}.png'.format(epoch, step))
            save_images(samples, [8, 8], samples_path)

if __name__ == '__main__':
    main()
