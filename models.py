'''
Created on Dec 24, 2016

A set of models used to evaluate street sign classification.

@author: jim
'''
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from support import conv2d, avgpool2d
from support import maxpool2d
import numpy as np



def LeNet(x, keep_prob): 
    '''
    Straight LeNet from the lab.
    '''   
    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    num_channels = x.get_shape()[3].value
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Activation.
    w1 = tf.Variable(tf.truncated_normal([5,5,num_channels,6],stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = conv2d(x, w1, b1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1)
    
    # Layer 2: Convolutional. Output = 10x10x16.
    w2 = tf.Variable(tf.truncated_normal([5,5,6,16],stddev=sigma))
    b2 = tf.Variable(tf.zeros(16))
    conv2 = conv2d(conv1, w2, b2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2)

    # Flatten. Input = 5x5x16. Output = 400.
    flattened_size = 5*5*16
    fc1 = tf.reshape(conv2, [-1, flattened_size])
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    w3 = tf.Variable(tf.truncated_normal([flattened_size, 120],stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc1,w3),b3)
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    w4 = tf.Variable(tf.truncated_normal([120, 84],stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,w4),b4)
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([84, 43],stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2,w5),b5)

    return logits

# 
def LeNet_dropout(x, keep_prob):
    '''
    Straight LeNet from the lab.
    '''   
    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    num_channels = x.get_shape()[3].value
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Activation.
    w1 = tf.Variable(tf.truncated_normal([5,5,num_channels,6],stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = conv2d(x, w1, b1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    
    # Layer 2: Convolutional. Output = 10x10x16.
    w2 = tf.Variable(tf.truncated_normal([5,5,6,16],stddev=sigma))
    b2 = tf.Variable(tf.zeros(16))
    conv2 = conv2d(conv1, w2, b2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2)

    # Flatten. Input = 5x5x16. Output = 400.
    flattened_size = 5*5*16
    fc1 = tf.reshape(conv2, [-1, flattened_size])
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    w3 = tf.Variable(tf.truncated_normal([flattened_size, 120],stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc1,w3),b3)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    w4 = tf.Variable(tf.truncated_normal([120, 84],stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,w4),b4)
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([84, 43],stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2,w5),b5, name="logits")

    return logits

def inception1(x, keep_prob): 
    '''
    A simple inception module that includes a 1x1 and 5x5 convolution.
    '''   
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    num_channels = x.get_shape()[3].value
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    w1 = tf.Variable(tf.truncated_normal([5,5,num_channels,6],stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = conv2d(x, w1, b1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1)
    # pad to 16 x 16 x 6
    conv1 = tf.pad(conv1, [[0,0],[1,1],[1,1],[0,0]])
    

    # 1x1: input = 32x32x3. Output = 32,32,x6
    w2 = tf.Variable(tf.truncated_normal([1,1,num_channels,6],stddev=sigma))
    b2 = tf.Variable(tf.zeros(6))
    conv2 = conv2d(x, w2, b2)
    
    # Pooling. Input = 28x28x6. Output = 16x16x6.
    conv2 = maxpool2d(conv2)

    # 2 14x14x6 to 14x14x12
    conv3 = tf.concat(3,[conv1, conv2])
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    # Flatten. Input = 14x14x12. Output = 3920
    flattened_size = 16*16*12
    fc1 = tf.reshape(conv3, [-1, flattened_size])
    
    # TODO: Layer 3: Fully Connected. Input = 3920. Output = 84.
    w3 = tf.Variable(tf.truncated_normal([flattened_size, 84],stddev=sigma))
    b3 = tf.Variable(tf.zeros(84))
    fc1 = tf.add(tf.matmul(fc1,w3),b3)
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    w4 = tf.Variable(tf.truncated_normal([84, 43],stddev=sigma))
    b4 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc1,w4),b4)

    return logits

def inception2(x, keep_prob): 
    '''
    The inception module that was used as an example in the class.
    '''   
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    num_channels = x.get_shape()[3].value
    
    # Average pooling...
    avg_pool = avgpool2d(x,k=1)
    
    # Layer 1: Convolutional. Input = 32x32x?. Output = 32x32x6.
    w1 = tf.Variable(tf.truncated_normal([1,1,num_channels,6],stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = conv2d(avg_pool, w1, b1)

    # 1x1: input = 32x32x?. Output = 32,32,x6
    w2 = tf.Variable(tf.truncated_normal([1,1,num_channels,6],stddev=sigma))
    b2 = tf.Variable(tf.zeros(6))
    conv2 = conv2d(x, w2, b2)
    
    # 1x1: input = 32x32x6. Output = 32,32,x6
    w3 = tf.Variable(tf.truncated_normal([3,3,6,6],stddev=sigma))
    b3 = tf.Variable(tf.zeros(6))
    conv3 = conv2d(conv2, w3, b3, padding='SAME')
    
    # 1x1: input = 32x32x6. Output = 32,32,x6
    w4 = tf.Variable(tf.truncated_normal([5,5,6,6],stddev=sigma))
    b4 = tf.Variable(tf.zeros(6))
    conv4 = conv2d(conv2, w4, b4, padding='SAME')
    
    # 3 14x14x6 to 14x14x24
    conv5 = tf.concat(3,[conv1, conv2, conv3, conv4])
    
    # Flatten. Input = 14x14x12. Output = 3920
    fc1 =flatten(conv5)
    flattened_size = fc1.get_shape()[1].value
    
    #Layer 3: Fully Connected. Input = 3920. Output = 400.
    w3 = tf.Variable(tf.truncated_normal([flattened_size, 400],stddev=sigma))
    b3 = tf.Variable(tf.zeros(400))
    fc1 = tf.add(tf.matmul(fc1,w3),b3)
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    #Layer 3: Fully Connected. Input = 400. Output = 84.
    w4 = tf.Variable(tf.truncated_normal([400, 84],stddev=sigma))
    b4 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,w4),b4)
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([84, 43],stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2,w5),b5)

    return logits



def multiscale1(x, keep_prob):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    num_channels = x.get_shape()[3].value
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # TODO: Activation.
    w1 = tf.Variable(tf.truncated_normal([5,5,num_channels,6],stddev=sigma))
    b1 = tf.Variable(tf.zeros(6))
    conv1 = conv2d(x, w1, b1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # TODO: Activation.
    w2 = tf.Variable(tf.truncated_normal([3,3,6,16],stddev=sigma))
    b2 = tf.Variable(tf.zeros(16))
    conv2 = conv2d(conv1, w2, b2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2)


    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = tf.concat(1, [flatten(conv2),flatten(conv1)])
    
    # TODO: Layer 3: Fully Connected. Input = flattened_size. Output = 84.
    w3 = tf.Variable(tf.truncated_normal([fc1.get_shape()[1].value, 84],stddev=sigma))
    b3 = tf.Variable(tf.zeros(84))
    fc1 = tf.add(tf.matmul(fc1,w3),b3)
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([84, 43],stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc1,w5),b5)

    return logits

