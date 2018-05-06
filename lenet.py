#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob, classes):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name = "conv_1_weights")
    conv1_b = tf.Variable(tf.zeros(6), name = "conv_1_bias")
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Layer 2: Convolutional. Output = 24x24x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name = "conv_2_weights")
    conv2_b = tf.Variable(tf.zeros(16), name = "conv_2_bias")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 24x24x16. Output = 12x12x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 2304.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 2304. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2304, 400), mean = mu, stddev = sigma), name = "fc1_weights")
    fc1_b = tf.Variable(tf.zeros(400), name = "fc1_bias")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    drop = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 120.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name = "fc2_weights")
    fc2_b  = tf.Variable(tf.zeros(120), name = "fc2_bias")
    fc2    = tf.matmul(drop, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 120. Output = classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, classes), mean = mu, stddev = sigma), name = "fc3_weights")
    fc3_b  = tf.Variable(tf.zeros(classes), name = "fc3_bias")
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
