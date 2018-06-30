import tensorflow as tf
import read_data as rd
import numpy as np
import pandas as pd
from pandas.io import sql
import  pymysql as PyMySQL
from sqlalchemy import create_engine


#sess = tf.InteractiveSession()
pd.set_option('display.width',1000)
restore = True
Train = False
checkpoint_file = "C:/tmp/tensclass/tensor.chk"
#(xData, yData) = rd.GenLogicRegData()
( Revdict, nxData, nyData, nzData ,df_in) = rd.Parse_data(Train)


input=12
output=35
print(nxData.shape)
print(nxData.size)
print(nzData.shape)
print(nzData.size)


#xData= nxData[:len(nxData)-len(nxData)%20].reshape(-1,20,input)
xData= nxData
print(xData.shape)
print(xData.size)
#yData= nyData[:len(nyData)-len(nyData)%20].reshape(-1,20,output)

#zData = np.array([([1, 0] if yEl == True else [0, 1]) for yEl in nzData])
zData=nzData
#norm_z= zData[:len(zData)-len(zData)%20].reshape(-1,20,output)
norm_z= zData
print(norm_z.shape)
print(norm_z.size)

num_periods =20

tf.reset_default_graph()

num_neurons=200
num_units =200
dropout =0.5
num_layers=2
x=tf.placeholder(tf.float32,[None,num_periods,input])
y_=tf.placeholder(tf.float32,[None,num_periods,output])

cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
    cell = tf.contrib.rnn.DropoutWrapper(
      cell, output_keep_prob=1.0 - dropout)
    cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)


#cell=tf.nn.rnn_cell.GRUCell(num_neurons)
#rnn_output,states =tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

## bidirectional RNN
rnn_output , states = tf.nn.bidirectional_dynamic_rnn(cell, cell,x,dtype=tf.float32)
print("rnn output")
print(rnn_output[0])
rnn_output=tf.concat([rnn_output[0],rnn_output[1]], axis=2)

stacked_rnn_output = tf.reshape(rnn_output,[-1,num_neurons*2])
#stacked_outputs = tf.layers.dense(stacked_rnn_output,output)
#print(stacked_outputs.shape)

weight = tf.Variable(tf.truncated_normal([num_neurons*2,output], stddev=0.01, dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1,shape=[output],dtype=tf.float32))


stacked_outputs = tf.matmul(stacked_rnn_output,weight)+bias
print(stacked_outputs.shape)


z = tf.reshape(stacked_outputs, [-1, num_periods,output])

#init = tf.global_variables_initializer()
print(z.shape)
z_ = tf.placeholder(tf.float32, [None, num_periods, output], name="z_")
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=z_, logits=z)

loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.FtrlOptimizer(0.4)
train_step_function = optimizer.minimize(loss)


def TrainWithBatch(steps, train_func, batch_size):
    init = tf.global_variables_initializer()
    dataset_size = len(xData)


    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()

        if restore:
            print("Loading variables from %s" % checkpoint_file)
            saver.restore(sess, checkpoint_file)
            inData=xData.reshape(-1,20,input)
            feed = {x: inData}
            #print(sess.run(tf.argmax(z, 1), feed_dict=feed))
            #print(sess.run(z, feed_dict=feed))

            #df_features = pd.DataFrame(data=xData)
            Onehot_Pred = sess.run(z, feed_dict=feed)
            Onehot_Pred = Onehot_Pred.reshape(-1,35)
            print(Onehot_Pred)
            Ind_Pred=sess.run(tf.argmax(Onehot_Pred, 1), feed_dict=feed)
            print(Ind_Pred)
            Ind_Pred = Ind_Pred.reshape(-1)
            print(Ind_Pred)
            print(df_in)
            symbols_out_pred=[]
            for i in range(0 ,len(Ind_Pred)):
                symbols_out_pred.append(Revdict[round(float(Ind_Pred[i]),3)])

            print(symbols_out_pred)
            Cols=[]
            for i in range(0 ,35):
                Cols.append(Revdict[i])
            # index = ['Row' + str(i) for i in range(1, len(output) + 1)]
            print(Cols)
            df_predout = pd.DataFrame(data=Onehot_Pred, columns=Cols)
            """["-1.2","2","3","4","5","6","7","8","9","10","11"
                                                            "11","12","13","14","15","16","17","18","19","20",
                                                            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                                                            "31", "32", "33", "34", "35"
                                                           ])
            """
            df_sympred=pd.DataFrame(data=symbols_out_pred, columns=["PRED"])
            df_predout = pd.concat([df_in, df_predout,df_sympred], axis=1)
            print(df_predout)
            cnx = create_engine('mysql+pymysql://RDSMUM:Sudhir123@rdsmum.cuzosyn2xlyq.ap-south-1.rds.amazonaws.com:3306/mumdb', echo=False)
            #sql.write_frame(df_predout, con=Conn, name='PREDOUT', if_exists='replace', flavor='mysql')
            df_predout.to_sql( con=cnx, name='PREDOUT', if_exists='replace', index=False)


            if Train:
                feed = {x: xData, z_: norm_z}
                #print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))
                df_out = pd.DataFrame (data=zData)
                #df_check =df_check.sort_values(["TIMESTAMP","PROBUP","PROBDN"],ascending=[True, True, True])
                correct_prediction = tf.equal(tf.argmax(z_, 2), tf.argmax(z, 2))
                verify =sess.run(correct_prediction ,feed_dict=feed)
                df_correct = pd.DataFrame(data=verify)
                df_check = pd.concat([df_in,df_out], axis=1)
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

                #result = sess.run(merged_summary, feed_dict=feed)
                #writer.add_summary(result, i)

                if (i+1) % 100 == 0:
                    print("After %d iterations :" % i)
                    #print ("W: (%f)" % sess.run(W[0][0]))
                    #print(sess.run([W1]))
                    #print("b: %f" % sess.run(b[0]))
                    #print("cross_entropy: %f" % sess.run(cross_entropy, feed_dict=feed))

                    correct_prediction = tf.equal(tf.argmax(z_, 2), tf.argmax(z, 2))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("Accuracy : %f " % sess.run(accuracy,feed_dict={x:xData , z_: norm_z}))
                    writer.close()

        print("saving Variables to %s" % checkpoint_file)
        saver.save(sess, checkpoint_file)

#TrainWithOnePoint(500, train_step_function )
batchsize=round(len(xData))
TrainWithBatch(3000,train_step_function,batchsize)
