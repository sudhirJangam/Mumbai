import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import os



random.seed(111)
rng=pd.date_range(start=2000, periods=209, freq='M')
ts=pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()

ts.head(5)

print(type(ts))

TS=np.array(ts)
num_periods =20
f_horizon =1
x_data=TS[:(len(TS)-(len(TS)%num_periods))]
x_batches=x_data.reshape(-1,20,1)
y_data=TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
y_batches =y_data.reshape(-1,20,1)
print(y_data.shape)
print(y_batches.shape)

testX = TS [-(num_periods+f_horizon):-1].reshape(-1,20,1)
testY = TS[-(num_periods):].reshape(-1,20,1)
tf.reset_default_graph()


input=1
output=1
num_neurons=100

x=tf.placeholder(tf.float32,[None,num_periods,input])
y=tf.placeholder(tf.float32,[None,num_periods,output])

cell=tf.nn.rnn_cell.GRUCell(num_neurons)
rnn_output,states =tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

print(rnn_output.shape)
stacked_rnn_output = tf.reshape(rnn_output,[-1,num_neurons])
#stacked_outputs = tf.layers.dense(stacked_rnn_output,output)
#print(stacked_outputs.shape)

weight = tf.Variable(tf.truncated_normal([num_neurons,output], stddev=0.01, dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1,shape=[output],dtype=tf.float32))


stacked_outputs = tf.matmul(stacked_rnn_output,weight)+bias
print(stacked_outputs.shape)


outputs = tf.reshape(stacked_outputs, [-1, num_periods,output])
print(outputs.shape)
loss = tf.reduce_sum(tf.square(outputs - y ))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

epocs =5000

with tf.Session() as sess:
      init.run()
      for ep in range(epocs):
          sess.run(training_op,feed_dict={x:x_batches, y:y_batches})
          if ep % 100 == 0:
              mse = loss.eval(feed_dict={x:x_batches, y:y_batches})
              print(ep, "MSE :",mse)

      y_pred=sess.run(outputs, feed_dict={x:testX})
      print(y_pred)
