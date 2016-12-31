'''
Created on Dec 23, 2016

@author: jim
'''
from LoadSignData import *


from sklearn.utils import shuffle
import tensorflow as tf
from models import LeNet, LeNet_dropout
from models import inception2
from models import multiscale1
from csv import DictWriter
from datetime import datetime
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
import time
from support import batch
from evaluate_image import load_image, get_images_to_evaluate

def run_fit(load_data, model, epochs=100, batch_size=128, learning_rate=0.001, 
            write_results=False, write_model=False):
    '''
    Execute a model fit using the data, model and parameters provided.
    '''
    start_time = time.time()
    
    X_train_o, y_train_o, X_test_o, y_test_o = load_data()
    
    # split the training data into training/validation sets
#     X_train, X_validation, y_train, y_validation = train_test_split(X_train_o, y_train_0, test_size=0.2)
    splitter = StratifiedShuffleSplit(y_train_o, n_iter=1, test_size=0.2)
    for train_index, validation_index in splitter:
        X_train = X_train_o[train_index]
        y_train = y_train_o[train_index]
        X_validation = X_train_o[validation_index]
        y_validation = y_train_o[validation_index]
    
    
    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    
    print()
    print("Data Source:    {}".format(load_data.__name__))
    print("Model:          {}".format(model.__name__))
    print("Image Shape:    {}".format(X_train[0].shape))
    print("Learning Rate:  {}".format(learning_rate))
    print("Epochs:         {}".format(epochs))
    print("Batch Size:     {}".format(batch_size))
    print("Training Set:   {} samples".format(len(X_train)))
    print("Test Set:       {} samples".format(len(X_validation)))
        
    X_train, y_train = shuffle(X_train, y_train)

    x = tf.placeholder(tf.float32, (None, X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2]))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32, (None))
    one_hot_y = tf.one_hot(y, 43)

    logits = model(x, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for batch_x, batch_y in batch(X_data, y_data, batch_size):
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples, loss
    
    # temporary: having problems loading saved session :-(
    def evaluate_images(image_names):
        sign_codes = read_sign_codes()
        
        sess = tf.get_default_session()
        images = rgb2gray([load_image(image_name) for image_name in image_names])
        probs = tf.nn.softmax(logits)
        
        result_probs = sess.run(probs, feed_dict={x: images, keep_prob: 1.0})
        for i in range(len(image_names)):
            print(image_names[i])
            pbs = result_probs[i]
            pbs_index = zip(range(len(pbs)), pbs)
            sorted_pbs = sorted(pbs_index, key=lambda pb: -pb[1])
            for v in sorted_pbs:
                print("{:>2}- {:50}: {:7.6}".format(v[0],sign_codes[str(v[0])],v[1]))
            

    max_accuracy = 0.0
    min_loss = 1000000000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print("Training...")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for batch_x, batch_y in batch(X_train, y_train, batch_size):
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
                
            validation_accuracy, loss = evaluate(X_validation, y_validation)
            dobreak = validation_accuracy - max_accuracy < 0.0001
            max_accuracy = np.max([max_accuracy, validation_accuracy])
            min_loss = np.min([min_loss, loss])
            print("EPOCH {}: Accuracy: {:>.4}; Loss: {:>7.4}".format(i+1,validation_accuracy, loss))
            if dobreak:
                break
        print("Done!")
        final_epoch = i+1
        
        if write_model:
            # I am having trouble saving/restoring a TF session, so I had to evaluate the
            # new images in-line here.
            evaluate_images(get_images_to_evaluate())
            
            try:
                saver
            except NameError:
                saver = tf.train.Saver()
            saver.save(sess, 'trafic_signs.ckpt')
            print("Model saved")
    
    if write_results:
        # record the results for posterity :)
        with open('results.csv','a') as results:
            writer = DictWriter(results,fieldnames = ['Date','Data Source',
                                                       'Model','Epochs','Batch Size','Learning Rate','Max Accuracy','Min Loss','Time (s)',
                                                       'Training Samples','Test Samples'])
            writer.writerow({'Date': str(datetime.now()),
                             'Data Source': load_data.__name__,
                             'Training Samples': len(X_train),
                             'Test Samples':len(X_validation),
                             'Model':model.__name__,
                             'Epochs':final_epoch,
                             'Learning Rate':learning_rate,
                             'Max Accuracy': "{:>.4}".format(max_accuracy),
                             'Min Loss':"{:.3}".format(min_loss),
                             'Batch Size': batch_size,
                             'Time (s)':"{:.3}".format(time.time()-start_time)})
    

if __name__ == '__main__':
    
    run_fit(data_greyscale_rotate, LeNet_dropout, learning_rate=0.001, batch_size=256, 
            write_results=False, write_model=True)
    
    if False:
        data_sources = [data, data_yuv_rotate, data_yuv, data_greyscale, data_greyscale_rotate]
        data_sources = [data_greyscale_rotate]
        models = [LeNet, LeNet_dropout, multiscale1,inception2]
        models = [LeNet_dropout, multiscale1]
        batch_sizes = [256]
        learning_rates = [0.001]*5
        
        for ds in data_sources:
            for md in models:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        run_fit(ds, md, batch_size=batch_size, learning_rate=learning_rate)
