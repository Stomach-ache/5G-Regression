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


print (tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# path to the training data
os.chdir("/")
for file in glob.glob("*.csv"):
    print(file)

    


inst, label = [], []
test_X, test_y = [], []

file_id = 0

for file in glob.glob("*.csv"):
    
    my_data = genfromtxt(file, delimiter=',')[1:]
    if file_id < 5:
        test_X = test_X + list(my_data[:, :-1])
        test_y = test_y + list(my_data[:, -1])
    else:
        inst = inst + list(my_data[:, :-1])
        label = label + list(my_data[:, -1])
    
    file_id += 1
    
train_X = np.array(inst)
train_y = np.array(label)
test_X = np.array(test_X)
test_y = np.array(test_y)


poly = PolynomialFeatures(2)
train_X = poly.fit_transform(train_X)

#min_max_scaler = preprocessing.MinMaxScaler()
#train_X = min_max_scaler.fit_transform(train_X)
train_X = preprocessing.scale(train_X)

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


batch_size = 2056
batch = Batch(train_X, train_y, batch_size)

input_size = train_X.shape[0]

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 171], name="myInput")
y = tf.placeholder(dtype = tf.int32, shape = [None, 1])

# Fully connected layer 
dense = tf.contrib.layers.fully_connected(x, 128, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dropout, 512, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

dense = tf.contrib.layers.fully_connected(dropout, 256, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)
    
dense = tf.contrib.layers.fully_connected(dropout, 128, tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5, training=True)

logits = tf.layers.dense(inputs=dropout, units=1)
logits = tf.identity(logits, name="myOutput")

# Define a loss function
loss = tf.losses.mean_squared_error(labels = y, predictions = logits)
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

for i in range(30000):
        #print('EPOCH', i)
        batch_x, batch_y = batch.getBatch()
        batch_y = batch_y.reshape(-1, 1)
        #print (batch_x.shape, batch_y.shape)
        #print (batch_x, batch_y)
        _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
        if i % 100 == 0:
            print(f"#{i} iteration, Loss: {loss_val}")
        #print('DONE WITH EPOCH')

#_logits = sess.run([logits], feed_dict={x: test_X})
#mse = tf.losses.mean_squared_error(labels = test_y, predictions = logits)



tf.saved_model.simple_save(sess,
            "./model_" + str(time.time()),
            inputs={"myInput": x},
            outputs={"myOutput": logits})


