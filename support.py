'''
Created on Dec 24, 2016

@author: jim
'''
import tensorflow as tf

def batch(X, y, batch_size):
    for offset in range(0, len(X), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X[offset:end], y[offset:end]
        yield batch_x, batch_y


def conv2d(x, W, b, strides=1,padding='VALID'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, padding='SAME'):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def avgpool2d(x, k=2, padding='SAME'):
    return tf.nn.avg_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

