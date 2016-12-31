'''
Created on Dec 29, 2016

@author: jim
'''
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from matplotlib.colors import Colormap

def view(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.show()

if __name__ == '__main__':
    images = ["no_right_turn",
              "stop",
              "no_parking",
              "speed_limit_30",
              "yellow_diamond",
              "yield",
              "dead_end",
              "keep_right"]
    for imgname in images:
        img = cv2.imread(imgname+".jpg", cv2.IMREAD_COLOR)
        res = cv2.resize(img,(32, 32), interpolation = cv2.INTER_CUBIC)
        view(res)
        cv2.imwrite(imgname+"_resized.jpg",res)
        pass
