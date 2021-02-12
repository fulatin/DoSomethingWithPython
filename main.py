import tensorflow as tf
import tensorflow.keras.datasets
import tensorflow.keras.layers
import tensorflow.keras.losses
import tensorflow.keras.optimizers
from tensorflow.keras import Sequential
import cv2
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense



data = tf.keras.datasets.cifar10.load_data()

def getDataBases(features,lables):
    return tf.data.Dataset.from_tensor_slices(features,lables)

def weightvariable(shape):
    init = tf.random.normal(shape)
    return tf.Variable(init)

def biasvariable(shape):
    init  =tf.random.normal(shape)
    return tf.Variable(init)

def maxpool(input):
    return tf.nn.max_pool2d(input,[1,2,2,1],[1,2,2,1],"SAME")


def Train():
    tf.compat.v1.disable_eager_execution()
    with tf.name_scope("data"):
        x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[-1,32,32,3])
        y_true = tf.compat.v1.placeholder(dtype=tf.float32,shape=[-1,10])

    with tf.name_scope("model"):
        with tf.name_scope("conv1"):
            con_w_1 = weightvariable([3,3,3,10])
            con_b_1 = biasvariable([10])
            con_x_1 =tf.nn.relu( tf.nn.conv2d(x,con_w_1,[1,1,1,1],"SAME")+con_b_1)

        with tf.name_scope("conv2"):
            con_w_2 = weightvariable([3,3,10,1])
            con_b_2  =biasvariable([1])
            con_x_2 = tf.nn.relu(tf.nn.conv2d(con_x_1,con_w_2,[1,1,1,1],"Same")+con_b_2)

        with tf.name_scope("maxpool1"):
            pool = maxpool(con_x_2)

        with tf.name_scope("conv3"):
            con_w_3 = weightvariable([3,3,1,1])
            con_b_3 = biasvariable([1])
            con_x_3 = tf.nn.relu(tf.nn.conv2d(pool,con_w_3,[1,1,1,1],"SAME")+con_b_3)

        with tf.name_scope("maxpool2"):
            pool2 = maxpool(con_x_3)
        
        with tf.name_scope("fullConnectLayer"):
            x_fc = tf.reshape(pool2,[-1,8*8])
            with tf.name_scope("layer1"):
                w_fc_1 = weightvariable([8*8,32])
                b_fc_1 = biasvariable([32])
                x_fc_1 =tf.nn.relu( tf.matmul(x_fc,w_fc_1)+b_fc_1)
            with tf.name_scope("outlayer"):
                w_fc_2 = weightvariable([32,10])
                b_fc_2 = biasvariable(10)
                y_pridict = tf.nn.softmax(tf.matmul(x_fc_1,w_fc_2)+b_fc_2)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(-y_true*tf.math.log(y_pridict))
        with tf.name_scope("train"):
            train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss)
        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as sess:
            dataset = getDataBases(data[0][0]/255,tf.one_hot(data[0][1],10))
            d = tf.data.Dataset.batch(dataset,10)
            
def test():
    dataset = getDataBases(data[0][0]/255,tf.one_hot(data[0][1],10))
    d = tf.data.Dataset.batch(dataset,10)
    print(d)
        



if __name__=="__main__":
    test()



