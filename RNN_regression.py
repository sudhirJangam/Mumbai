import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import read_data as rd
import os



restore = False
Train = True
checkpoint_file = "c:/tmp/tensreg/tensor_reg.chk"
#(xData, yData) = rd.GenLogicRegData()
(nxData, nyData, zData ,df_in) = rd.Parse_data(Train)

print(df_in)
#print(nxData)


input=2
output=1
xData= nxData[:len(nxData)-len(nxData)%20].reshape(-1,20,input)
print(xData)
yData= nyData[:len(nyData)-len(nyData)%20].reshape(-1,20,output)
num_periods =20

tf.reset_default_graph()


num_neurons=100

x=tf.placeholder(tf.float32,[None,num_periods,input])
y_=tf.placeholder(tf.float32,[None,num_periods,output])

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
loss = tf.reduce_sum(tf.square(outputs - y_ ))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step_function = optimizer.minimize(loss)
#init = tf.global_variables_initializer()



def TrainWithBatch(steps, train_func, batch_size):
    init = tf.global_variables_initializer()
    dataset_size = len(xData)


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
            output = sess.run(y, feed_dict=feed)
            # index = ['Row' + str(i) for i in range(1, len(output) + 1)]
            df_predout = pd.DataFrame(data=output, columns=["PREDICTION"])
            df_predout = pd.concat([df_in, df_predout], axis=1)
            print(df_predout)

            if Train:
                feed = {x: xData, y_:yData}
                print("loss: %f" % sess.run(loss, feed_dict=feed))
                df_out = pd.DataFrame (data=yData, columns=["ACTUAL"])
                #df_check =df_check.sort_values(["TIMESTAMP","PROBUP","PROBDN"],ascending=[True, True, True])
                difference = y - y_
                diff =sess.run(difference ,feed_dict=feed)
                df_diff = pd.DataFrame(data=diff , columns=["DIFF"])
                df_check = pd.concat([df_predout, df_out,df_diff], axis=1)
                df_check["CHG"]=df_check["DIFF"]/df_check["CLOSE"]
                df_check = df_check.sort_values(["PREDICTION","TIMESTAMP"],ascending=[True,True])
                #df_check = df_check[(df_check["SYMBOL"]=="GMRINFRA")]
                print(df_check)

                df_plot =pd.concat([df_predout,df_out],axis=1)
                df_plot=df_plot[["TIMESTAMP","SYMBOL","PREDICTION","ACTUAL"]]
                #df_plot=df_plot[(df_plot["SYMBOL"]=="NHPC")]
                """
                #plt.xticks(x,df_plot["TIMESTAMP"])
                plt.plot(df_plot["TIMESTAMP"],df_plot["ACTUAL"], label="ACTUAL")
                plt.plot(df_plot["TIMESTAMP"], df_plot["PREDICTION"],   label="PREDICTION" )
                plt.legend(["ACTUAL", "PREDICTION"], loc=4)
                #df_plot.plot(style=['o', 'rx'])
                plt.show()
                """


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

                batch_xs = xData[batch_start_idx: batch_end_idx]
                batch_ys = yData[batch_start_idx: batch_end_idx]

                feed = {x: batch_xs, y_: batch_ys}
                sess.run(train_func, feed_dict=feed)

                #result = sess.run(merged_summary, feed_dict=feed)
                #writer.add_summary(result, i)

                if (i+1) % 100 == 0:
                    print("After %d iterations :" % i)
                    # print ("W: (%f)" % sess.run(W[0]))
                    # print( sess.run( [W]))
                    # print(sess.run(b))
                    print("loss: %f" % sess.run(loss, feed_dict=feed))
                    writer.close()

        print("saving Variables to %s" % checkpoint_file)
        saver.save(sess, checkpoint_file)

#TrainWithOnePoint(500, train_step_function )
batchsize=round(len(xData))
TrainWithBatch(5000,train_step_function,batchsize)
