import tensorflow as tf
import read_data as rd
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


restore = True
Train = True
checkpoint_file = "c:/tmp/tensreg/tensor_reg.chk"
#(xData, yData) = rd.GenLogicRegData()
(xData, yData, zData ,df_in) = rd.Parse_data(Train)

print(df_in)
print(xData)
W = tf.Variable(tf.random_normal(([23,40]), stddev=0.1))
#W = tf.Variable(tf.zeros([23, 12]) , name="W")
#W1 = tf.Variable(tf.zeros([30, 20]) , name="W1")
W1 = tf.Variable(tf.random_normal(([40,10]), stddev=0.1))
W2 = tf.Variable(tf.random_normal(([10,1]), stddev=0.1))
print(W)
b=tf.Variable(tf.random_normal(shape=[40],stddev=0.1))
b1=tf.Variable(tf.random_normal(shape=[10], stddev=0.1))
b2=tf.Variable(tf.random_normal(shape=[1], stddev=0.1))

#b=tf.Variable(tf.constant(0.1,shape=[40]))
#b1=tf.Variable(tf.constant(0.1,shape=[20]))
#b2=tf.Variable(tf.constant(0.1,shape=[1]))
#b = tf.Variable(tf.ones([12]) , name="b")
#b1 = tf.Variable(tf.ones([1]) , name="b1")


x = tf.placeholder(tf.float32, [None, 23], name="x")
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

#x_hist=tf.summary.histogram("x",x)
Wx = tf.matmul(x, W)
y1= tf.nn.elu(Wx+b)

Wx1=tf.matmul(y1,W1)

#Wx1=tf.matmul(Wx,W1)
y2 = tf.nn.elu(Wx1+b1)
#h_fc1_drop = tf.nn.dropout(y2, 0.5)
Wx2=tf.matmul(y2,W2)
y = Wx2+b2
#h_fc2_drop = tf.nn.dropout(y3, 0.5)
#z =  Wx1+b1

W_hist = tf.summary.histogram("Weights", W )
b_hist = tf.summary.histogram("bias", b )
y_hist = tf.summary.histogram("y", y)


cost = tf.reduce_mean(tf.square(y_ - y))

train_step_function = tf.train.FtrlOptimizer(0.3).minimize(cost)
cost_hist = tf.summary.histogram("Cost", cost)



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
                print("cost: %f" % sess.run(cost, feed_dict=feed))
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

                #cnx = create_engine('mysql+pymysql://spark:sparkPass@13.126.33.137:3306/sparkml', echo=False)
                """
                conn = (pymysql.connect(host = '13.126.33.137',
                                        port = 3306,
                                        user = "spark",
                                        password = "sparkPass",
                                        database = "sparkml",
                                        charset="utf8"))

                #df_check.write_frame( con=conn, name='Office_RX', if_exists='replace', flavor='mysql')
                """
                #df_check.to_sql(name='sample_table', con=cnx, if_exists='append', index=False)
                #cnx.close()



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

                result = sess.run(merged_summary, feed_dict=feed)
                writer.add_summary(result, i)

                if (i+1) % 100 == 0:
                    print("After %d iterations :" % i)
                    # print ("W: (%f)" % sess.run(W[0]))
                    # print( sess.run( [W]))
                    # print(sess.run(b))
                    print("cost: %f" % sess.run(cost, feed_dict=feed))
                    writer.close()

        print("saving Variables to %s" % checkpoint_file)
        saver.save(sess, checkpoint_file)

#TrainWithOnePoint(500, train_step_function )
batchsize=round(len(xData))
TrainWithBatch(20000,train_step_function,batchsize)
