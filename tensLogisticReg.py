import tensorflow as tf
import read_data as rd
import numpy as np
import pandas as pd


restore = False
Train = True
checkpoint_file = "c:/tmp/tensclass/tensor.chk"
#(xData, yData) = rd.GenLogicRegData()
(xData, yData, zData ,df_in) = rd.Parse_data(Train)

print(df_in)
print(xData)
#W = tf.Variable(tf.zeros([18, 4]) , name="W")
W = tf.Variable(tf.random_normal(([28, 40]), stddev=0.1))
#W1 = tf.Variable(tf.zeros([4, 2]) , name="W1")
W1 = tf.Variable(tf.random_normal(([40,20]), stddev=0.1))
W2 = tf.Variable(tf.random_normal(([20,4]), stddev=0.1))
W3 = tf.Variable(tf.random_normal(([4,2]), stddev=0.1))
#W2 = tf.Variable(tf.zeros([2, 2]) , name="W2")
print(W)

b=tf.Variable(tf.random_normal(shape=[40],stddev=0.1))
b1=tf.Variable(tf.random_normal(shape=[20], stddev=0.1))
b2=tf.Variable(tf.random_normal(shape=[4], stddev=0.1))
b3=tf.Variable(tf.random_normal(shape=[1], stddev=0.1))

#b=tf.Variable(tf.constant(0.1,shape=[40]))
#b1=tf.Variable(tf.constant(0.1,shape=[20]))
#b2=tf.Variable(tf.constant(0.1,shape=[4]))
#b3=tf.Variable(tf.constant(0.1,shape=[2]))
#b = tf.Variable(tf.ones([18]) , name="b")
#b1 = tf.Variable(tf.ones([2]) , name="b1")
#b2 = tf.Variable(tf.ones([2]) , name="b2")
print(b1)


x = tf.placeholder(tf.float32, [None, 28], name="x")

#x_hist=tf.summary.histogram("x",x)
Wx = tf.matmul(x, W)
y1=tf.nn.elu(Wx+b)
Wx1=tf.matmul(y1,W1)

#Wx1=tf.matmul(Wx,W1)
y2 = tf.nn.elu(Wx1+b1)
h_fc1_drop = tf.nn.dropout(y2, 0.5)
Wx2=tf.matmul(h_fc1_drop,W2)
y3 = tf.nn.elu(Wx2+b2)
#h_fc2_drop = tf.nn.dropout(y3, 0.5)
#z =  Wx1+b1
Wx3=tf.matmul(y3,W3)

z=tf.nn.softmax(Wx3+b3)


W_hist = tf.summary.histogram("Weights", W )
b_hist = tf.summary.histogram("bias", b )
y_hist = tf.summary.histogram("y", z)
z_ = tf.placeholder(tf.float32, [None, 2], name="z_")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_, logits=z))

cost_hist = tf.summary.histogram("Cost", cross_entropy)
train_step_function = tf.train.FtrlOptimizer(0.4).minimize(cross_entropy)


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
            feed = {x: xData}
            #print(sess.run(tf.argmax(z, 1), feed_dict=feed))
            #print(sess.run(z, feed_dict=feed))

            df_features = pd.DataFrame(data=xData)
            output = sess.run(z, feed_dict=feed)
            # index = ['Row' + str(i) for i in range(1, len(output) + 1)]
            df_predout = pd.DataFrame(data=output, columns=["PROBUP", "PROBDN"])
            df_predout = pd.concat([df_in, df_predout], axis=1)
            print(df_predout)

            if Train:
                feed = {x: xData, z_: norm_z}
                print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))
                df_out = pd.DataFrame (data=zData)
                #df_check =df_check.sort_values(["TIMESTAMP","PROBUP","PROBDN"],ascending=[True, True, True])
                correct_prediction = tf.equal(tf.argmax(z_, 1), tf.argmax(z, 1))
                verify =sess.run(correct_prediction ,feed_dict=feed)
                df_correct = pd.DataFrame(data=verify)
                df_check = pd.concat([df_predout, df_out,df_correct], axis=1)
                print(df_check)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Accuracy : %f " % sess.run(accuracy, feed_dict={x: xData, z_: norm_z}))
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

                feed = { x: batch_xs, z_: batch_zs }
                sess.run(train_func, feed_dict=feed)

                result = sess.run(merged_summary, feed_dict=feed)
                writer.add_summary(result, i)

                if (i+1) % 100 == 0:
                    print("After %d iterations :" % i)
                    #print ("W: (%f)" % sess.run(W[0][0]))
                    #print(sess.run([W1]))
                    #print("b: %f" % sess.run(b[0]))
                    print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))

                    correct_prediction = tf.equal(tf.argmax(z_, 1), tf.argmax(z, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("Accuracy : %f " % sess.run(accuracy,feed_dict={x:xData , z_: norm_z}))
                    writer.close()

        print("saving Variables to %s" % checkpoint_file)
        saver.save(sess, checkpoint_file)

#TrainWithOnePoint(500, train_step_function )
batchsize=round(len(xData))
TrainWithBatch(20000,train_step_function,batchsize)
