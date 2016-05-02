import os
import hashlib

import scipy.misc
import numpy as np

def one_hot(labels, num_classes=10):
    num_samples = labels.shape[0]
    ones = np.zeros((num_samples, num_classes))
    ones[np.arange(num_samples), labels] = 1
    return ones

def save_images(images, size, path):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(path, img)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
