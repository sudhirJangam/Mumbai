import tensorflow as tf
import numpy as np
import pandas as pd
#import read_data as rd

def init_weights(shape):
    """ weight initialization"""
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forwardprop(X, w_1, w_2, w_11):
    """" forward propogration last layer is not softmax as softmax_cross_entropy_with_logits does it internally"""
    #h=tf.nn.sigmoid(tf.matmul(X, w_1))
    h = tf.nn.relu(tf.matmul(X, w_1))
    #h = tf.nn.softmax(tf.matmul(X, w_1))
    h1 = tf.nn.relu(tf.matmul(h, w_2))
    yh = tf.nn.relu(tf.matmul(h1, w_11))
    return yh

def create_data():
    dfile = "C:/tmp/data"
    df = pd.read_csv(dfile, sep="\t", usecols=[0, 11, 12, 13, 14], names=["Symbol", "em9", "sst1", "sst2", "Date"],
                         header=1)
    data = df.as_matrix(columns=df.columns[1:3])
    print(data)
    #target = np.concatenate(df.as_matrix(columns=df.columns[:1]))
    #print(target)

    #prepare column of 1s for bias

    N, M = data.shape
    all_X = np.ones((N, M+1))
    all_X[ :, 1:] = data

    #convert into one-hot vector

    num_labels = len(df.Symbol.unique())
    print(num_labels)
    labels = pd.DataFrame(df.Symbol.unique(), columns=["Symbol"])
    print(labels)
    onehot=pd.DataFrame(np.eye(num_labels))
    target = pd.concat([onehot,labels] , axis=1)

    onehotTarget = pd.merge(df, target, on="Symbol")
    all_Y = onehotTarget.as_matrix(columns=onehotTarget.columns[-num_labels:])
    print(all_Y)
    #return train_test_split(all_X, all_Y, test_size=0.33, randomw_state=RANDOM_SEED)
    return (all_X, all_X, all_Y,  all_Y)



def main():

    train_X, test_X, train_Y, test_Y = create_data()
    print(train_X)
    print(train_Y)

    # Layer's sizes
    x_size = train_X.shape[1]
    h_size = 20
    y_size = train_Y.shape[1]

    #Symbols
    x = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    #weight initialize
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights ((h_size, h_size))
    w_11 = init_weights((h_size, y_size))

    #forward propogation
    y_ = forwardprop(x, w_1, w_2, w_11)
    predict = tf.argmax(y_, axis=1)

    #backward propogation
    cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y , logits= y_))
    updates = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(100):

            for i in range (2):
                sess.run(updates, feed_dict={x: train_X[i :len(train_X)], y: train_Y[i : len(train_Y)]})
                Pred_out = sess.run(predict, feed_dict={x:train_X, y:train_Y})

                train_accuracy = sess.run(tf.reduce_mean(tf.cast((tf.argmax(y, axis=1) == Pred_out ), tf.float32)))

                print("Epoch = %d, train accuracy = %.2f%% " %(epoch +1 ,  train_accuracy*100.0 ))
                print(sess.run([w_1]))

    sess.close()


main()
#if _name_ == '_main_':
#    main()