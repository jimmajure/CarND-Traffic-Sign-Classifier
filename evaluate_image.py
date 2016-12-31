'''
Created on Dec 29, 2016

@author: jim
'''
import tensorflow as tf
import matplotlib.image as mpimg
from LoadSignData import rgb2gray
from models import LeNet_dropout
import cv2

def get_images_to_evaluate():
    return ["stop_resized.jpg","no_parking_resized.jpg","no_right_turn_resized.jpg",
            "yield_resized.jpg","yellow_diamond_resized.jpg","dead_end_resized.jpg",
            "speed_limit_30_resized.jpg","keep_right_resized.jpg"]

def evaluate_image(image, model):
    sess = tf.get_default_session()
    assert image.shape == (1,32,32,1)
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    keep_prob = tf.placeholder(tf.float32, (None))

    logits = model(x, keep_prob)
    probs = tf.nn.softmax(logits)
    
    result_probs = sess.run(probs, feed_dict={x: image, keep_prob: 1.0})
    print(result_probs)


def load_image(imagename):
    print(imagename)
    img = cv2.imread(imagename, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    evaluate(rgb2gray([load_image("stop_resized.jpg")]), LeNet_dropout, "traffic_signs.ckpt")
    
    