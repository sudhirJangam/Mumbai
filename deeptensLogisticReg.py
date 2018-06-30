import tensorflow as tf
import read_data as rd
import numpy as np

restore = False
Train = True
checkpoint_file = "c:/tmp/tensordeep.chk"
#(xData, yData) = rd.GenLogicRegData()
#(xData, yData, zData) = rd.Parse_mum()
(xData, yData, zData ,df_in) = rd.Parse_mum(Train)

print(xData.shape)

x = tf.placeholder(tf.float32, [None, 36], name="x")
z_ = tf.placeholder(tf.float32, [None, 2], name="z_")

x_image = tf.reshape(x, [-1, 9, 4, 1], name="image")
print(x_image)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d( x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1] , padding="SAME")
    #return tf.nn.conv1d(x,W,stride=2, padding="SAME")



def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")



W_conv1 = weight_variable([6,4,1,8])
b_conv1 = bias_variable([8])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2d(h_conv1)

W_conv2 = weight_variable([6,4,8,16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2d(h_conv2)


#fully connected layer
w_fc1 = weight_variable([48, 256])
b_fc = bias_variable([256])
h_pool2_flat = tf.reshape(h_pool2, [-1, 48])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1))

#drop out some of the neurons

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
w_fc2=weight_variable([256, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2



W_hist = tf.summary.histogram("Weights", w_fc2 )
b_hist = tf.summary.histogram("bias", b_fc2 )
y_hist = tf.summary.histogram("y", y_conv)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_, logits=y_conv))

cost_hist = tf.summary.histogram("Cost", cross_entropy)
train_step_function = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


def TrainWithBatch(steps, train_func, batch_size):
    init = tf.global_variables_initializer()
    dataset_size = len(xData)
    norm_z = np.array([([1, 0] if yEl == True else [0, 1]) for yEl in zData])

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        if restore:
            print("Loading variables from %s" % checkpoint_file)
            saver.restore(sess, checkpoint_file)
            feed = {x: xData , z_: norm_z, keep_prob: 0.5}
            print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))
            #print(sess.run(tf.argmax(y_conv, 1), feed_dict=feed))
            #print(sess.run(y_conv, feed_dict=feed))
            correct_prediction = tf.equal(tf.argmax(z_, 1), tf.argmax(y_conv, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy : %f " % sess.run(accuracy, feed_dict={x: xData, z_: norm_z, keep_prob: 0.5}))
        else:

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
                batch_zs = norm_z[batch_start_idx: batch_end_idx]

                feed = { x: batch_xs, z_: batch_zs , keep_prob:0.5 }
                sess.run(train_func, feed_dict=feed)

                result = sess.run(merged_summary, feed_dict=feed)
                writer.add_summary(result, i)

                if (i+1) % 100 == 0:
                    print("After %d iterations :" % i)
                    #print ("W: (%f)" % sess.run(W[0][0]))
                    #print(sess.run([W1]))
                    #print("b: %f" % sess.run(b[0]))
                    print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))

                    correct_prediction = tf.equal(tf.argmax(z_, 1), tf.argmax(y_conv, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("Accuracy : %f " % sess.run(accuracy,feed_dict={x:xData , z_: norm_z, keep_prob:0.5}))
                    writer.close()

        print("saving Variables to %s" % checkpoint_file)
        saver.save(sess, checkpoint_file)

#TrainWithOnePoint(500, train_step_function )
batchsize=round(len(xData))
TrainWithBatch(8000,train_step_function,batchsize)
