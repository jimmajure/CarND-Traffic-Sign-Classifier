'''
Created on Dec 24, 2016

@author: jim
'''
# Load pickled data
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader
from collections import Counter
import cv2



Yscale = [0.299, 0.587, 0.114]
Uscale = [-0.14713, -0.28886, 0.436]
Vscale = [0.615,-0.51499, -0.10001]

YUVscale = [Yscale, Uscale, Vscale]

def read_sign_codes():
    result = {}
    with open("signnames.csv") as f:
        reader = DictReader(f)
        for l in reader:
            result[l['ClassId']] = l['SignName']
    return result
    
def data():
    training_file = 'traffic-signs-data/train.p'
    testing_file = 'traffic-signs-data/test.p'
    
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_test, y_test

def rgb2gray(rgb):
    return np.reshape(np.dot(rgb, Yscale),[len(rgb),32,32,1])

def rgb2yuv(rgb):
    return np.dot(rgb, YUVscale)
    
def random_rotate(img):
    angle = random.randint(-15,15)
    scale = 1.* random.randint(900,1100)/1000.
    x_shift = random.randint(0,3)
    y_shift = random.randint(0,3)
    M = cv2.getRotationMatrix2D((15+x_shift,15+y_shift),angle, scale)
    return cv2.warpAffine(img, M, (32,32))
    
def data_standardized_128():
    X_train, y_train, X_test, y_test = data()
    X_train = 1.0*(X_train-128)/128
    X_test = 1.0*(X_test-128)/128
    return X_train, y_train, X_test, y_test

def data_greyscale():
    X_train, y_train, X_test, y_test = data()
    X_train = rgb2gray(X_train)
    X_test = rgb2gray(X_test)
    X_train = (X_train - np.mean(X_train)/np.std(X_train))
    X_test = (X_test - np.mean(X_test)/np.std(X_test))
    return X_train, y_train, X_test, y_test
    
def data_greyscale_rotate():
    X_train, y_train, X_test, y_test = data_yuv_rotate()
    Y_channel_train = X_train[:,:,:,0:1]
    Y_channel_test = X_test[:,:,:,0:1]
    
    return Y_channel_train, y_train, Y_channel_test, y_test
    
def data_yuv():
    '''
    Get the data and convert to YUV.
    Normalize the Y channel by its mean/std dev.
    '''
    X_train, y_train, X_test, y_test = data()
    X_train = rgb2yuv(X_train)
    X_test = rgb2yuv(X_test)
    
    Y_channel = X_train[:,:,:,0:1]
    Y_channel = (Y_channel - np.mean(Y_channel))/np.std(Y_channel)
    X_train = np.concatenate((Y_channel,X_train[:,:,:,1:3]), axis=3)
     
    Y_channel = X_test[:,:,:,0:1]
    Y_channel = (Y_channel - np.mean(Y_channel))/np.std(Y_channel)
    X_test = np.concatenate((Y_channel,X_test[:,:,:,1:3]), axis=3)
    
    return X_train, y_train, X_test, y_test
    
    
def data_yuv_rotate():
    '''
    Get the data and convert to YUV.
    Normalize the Y channel by its mean/std dev.
    '''
    X_train, y_train, X_test, y_test = data()
    X_train = rgb2yuv(X_train)
    X_test = rgb2yuv(X_test)
    
    Y_channel = X_train[:,:,:,0:1]
    Y_channel = (Y_channel - np.mean(Y_channel))/np.std(Y_channel)
    X_train = np.concatenate((Y_channel,X_train[:,:,:,1:3]), axis=3)
     
    Y_channel = X_test[:,:,:,0:1]
    Y_channel = (Y_channel - np.mean(Y_channel))/np.std(Y_channel)
    X_test = np.concatenate((Y_channel,X_test[:,:,:,1:3]), axis=3)
    
    X_train_out = np.concatenate((X_train, np.array([random_rotate(img) for img in X_train])), axis=0)
    y_train_out = np.concatenate((y_train, y_train), axis=0)
                             
    X_train_out = np.concatenate((X_train_out, np.array([random_rotate(img) for img in X_train])), axis=0)
    y_train_out = np.concatenate((y_train_out, y_train), axis=0)
                             
    
    return X_train_out, y_train_out, X_test, y_test
    
def view(X, y, cnt=10, clss=None):
    sign_codes = read_sign_codes()
    local_data = list(zip(X,y))
    if type:
        local_data = [v for v in local_data if v[1] == clss]
    for i in range(cnt):
        index = random.randint(0, len(local_data))
        image = local_data[index][0].squeeze()
        
        plt.figure(i)
        plt.title("{0}: {1}".format(local_data[index][1],sign_codes[str(local_data[index][1])]))
        plt.imshow(image)
    
    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data()
        
    
    data = Counter()
    for y in y_train:
        data[y] += 1
    
    sign_codes = read_sign_codes()
    for code in sign_codes.keys():
        print("{0:3}: {1:50} {2:5}".format(code, sign_codes[code], data[int(code)]))
    
    print("Total classes: {0}".format(len(sign_codes)))
    view(X_train, y_train, 10, 17)

