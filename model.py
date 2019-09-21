# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import glob, os
from numpy import genfromtxt
from tensorflow import keras
import time

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

from features import cal_features, cal_pcrr

print (tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# path to the training data
#os.chdir("./train_set")




with open("./data_new.csv") as file:
    my_data = pd.read_csv(file).to_numpy()
    #my_data = genfromtxt(file, delimiter=',')[1:]
print (my_data.shape)
print (type(my_data))

my_data = my_data[-1000000:]

train_X = np.array(my_data[:-10000, :-1])
train_y = np.array(my_data[:-10000, -1])
test_X = np.array(my_data[-10000:, :-1])
test_y = np.array(my_data[-10000:, -1])
test_y = test_y.reshape(-1, 1)



for f in cal_features(train_X):
    train_X = np.column_stack((train_X, f))

for f in cal_features(test_X):
    test_X = np.column_stack((test_X, f))

print (train_X.shape)

train_X = train_X[:, [1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]]
test_X = test_X[:, [1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]]

#poly = PolynomialFeatures(2)
# train_X = poly.fit_transform(train_X)

#min_max_scaler = preprocessing.MinMaxScaler()
#train_X = min_max_scaler.fit_transform(train_X)
train_X = preprocessing.scale(train_X)
test_X = preprocessing.scale(test_X)

print (train_y.shape, train_X.shape, test_X.shape, test_y.shape)


class Batch(object):
  def __init__(self, X, y, batch_size):
    self.batch_size = batch_size
    self.X = X
    self.y = y
    self.size = X.shape[0]
  def getBatch(self):
    indices = np.random.choice(range(self.size), self.batch_size)
    return self.X[indices], self.y[indices]


batch_size = 2046 * 4
batch = Batch(train_X, train_y, batch_size)


# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 14], name="myInput")
y = tf.placeholder(dtype = tf.float32, shape = [None, 1])

# Fully connected layer
dense = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu)
#dense = tf.contrib.layers.fully_connected(dense, 512, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)

#dense = tf.contrib.layers.fully_connected(dense, 1024, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dense, 512, tf.nn.relu)
#dense = tf.contrib.layers.fully_connected(dense, 128, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dense, 256, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)

#dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)
#
#dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)
#
#dense = tf.contrib.layers.fully_connected(dropout, 128, tf.nn.relu)
#dropout = tf.layers.dropout(
#    inputs=dense, rate=0.5, training=True)

logits = tf.layers.dense(inputs=dense, units=1)
logits = tf.identity(logits, name="myOutput")

# Define a loss function
loss = tf.losses.mean_squared_error(labels = y, predictions = logits)
#tmp_ = tf.reduce_mean(tf.nn.relu(-(tf.reshape(y, [-1]) + 103) * (tf.reshape(logits, [-1]) + 103)))

#loss = loss + 0.01 * tmp_
#loss = tf.losses.mean_squared_error(labels = y, predictions = logits)
#loss = tf.reduce_mean(tf.square(logits - y))
# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# Define an accuracy metric
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

print (type(train_X), train_X.shape)

  # Add a scalar summary for the snapshot loss.
tf.summary.scalar('loss', loss)

for i in range(20000):
        #print('EPOCH', i)
        batch_x, batch_y = batch.getBatch()
        batch_y = batch_y.reshape(-1, 1)
            #print (batch_x.shape, batch_y.shape)
        #print (batch_x, batch_y)
        _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
        if i % 100 == 0:
            validation, y_pred = sess.run([loss, logits], feed_dict={x:test_X, y: test_y})
            with open("test_y", "w") as f:
                f.write(" ".join([str(item) for item in test_y.ravel()]))
            with open("y_pred", "w") as f:
                f.write(" ".join([str(item) for item in y_pred.ravel()]))
            print(test_y)
            print(y_pred)
            test_y = test_y[::-1]
            pcrr = cal_pcrr(test_y, y_pred)
            print(f"#{i} iteration, Loss: {loss_val}, Validation: {validation}, pcrr: {pcrr}")
        #print('DONE WITH EPOCH')

#_logits = sess.run([logits], feed_dict={x: test_X})
#mse = tf.losses.mean_squared_error(labels = test_y, predictions = logits)



tf.saved_model.simple_save(sess,
            "./model_" + str(time.time()),
            inputs={"myInput": x},
            outputs={"myOutput": logits})


#test