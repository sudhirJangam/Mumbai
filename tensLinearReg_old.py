import tensorflow as tf
import read_data as rd
import numpy as np

(xData, yData, zData, df) = rd.Parse_mum(True)

print(xData)
print(yData)
#W = tf.random_normal(([2, 1]), stddev=0.1 )
#W1 = tf.random_normal(([1, 1]), stddev=0.1)
W = tf.Variable(tf.zeros([60, 36]) , name="W")
W1 = tf.Variable(tf.zeros([36, 1]) , name="W1")
print(W)
b = tf.Variable(tf.ones([36]) , name="b")
b1 = tf.Variable(tf.ones([1]) , name="b1")
print(b)


x = tf.placeholder(tf.float32, [None, 60], name="x")

#x_hist=tf.summary.histogram("x",x)
Wx = tf.matmul(x, W)
y1= Wx+b
Wx1=tf.matmul(y1,W1)
#Wx1=tf.matmul(Wx,W1)
y = Wx1+b1


W_hist = tf.summary.histogram("Weights", W )
b_hist = tf.summary.histogram("bias", b )
y_hist = tf.summary.histogram("y", y)
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

cost = tf.reduce_mean(tf.square(y_ - y))
#cost_hist = tf.summary.histogram("Cost", cost)
#train_step_function = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
train_step_function = tf.train.FtrlOptimizer(1).minimize(cost)


def TrainWithOnePoint(steps, train_func):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('C:/tmp/logfile', sess.graph)
        for i in range(steps):
            xs = xData[i % len(xData)]
            ys = yData[i % len(yData)]
            feed = { x: xs, y_: ys }
            sess.run(train_func, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            if (i+1) % 100 == 0:
                print("After %d iterations :" % i)
                print ("W: %f" % sess.run(W))
                print("b: %f" % sess.run(b))
                print("cost: %f" % sess.run(cost, feed_dict=feed))

        writer.close()


def TrainWithBatch(steps, train_func, batch_size):
    init = tf.global_variables_initializer()
    dataset_size = len(xData)
    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('C:/tmp/logfile', sess.graph)
        for i in range(steps):
            if dataset_size == batch_size:
                batch_start_idx=0
            elif dataset_size < batch_size:
                raise ValueError("data size %d is smaller than batch size %d" % (dataset_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % dataset_size

            batch_end_idx = batch_start_idx + batch_size

            batch_xs =  xData[batch_start_idx : batch_end_idx ]
            batch_ys = yData[batch_start_idx: batch_end_idx]

            feed = { x: batch_xs, y_: batch_ys }
            sess.run(train_func, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            if (i+1) % 100 == 0:
                print("After %d iterations :" % i)
               # print ("W: (%f)" % sess.run(W[0]))
                #print( sess.run( [W]))
                #print(sess.run(b))
                print("cost: %f" % sess.run(cost, feed_dict=feed))

        writer.close()

#TrainWithOnePoint(500, train_step_function )
dataset_size = round(len(xData))
TrainWithBatch(10000,train_step_function,dataset_size)
