import PIL.Image
import tensorflow as tf
import cv2
import tensorflow.keras.layers
import tensorflow.keras.losses
import tensorflow.keras.optimizers
from tensorflow import keras

# def getDataBases(features,lables):
#     return tf.data.Dataset.from_tensor_slices((features,lables))
# data = tf.keras.datasets.mnist.load_data()
# dataset = getDataBases(data[0][0]/255,tf.one_hot(data[0][1],10))
# def getx():
#     d = dataset.batch(10)
#     return d.as_numpy_iterator().next()[0]

# def gety():
#     d = dataset.batch(10)
#     return d.as_numpy_iterator().next()[1]

# def weightvariable(shape):
#     init = tf.random.normal(shape)
#     return tf.Variable(init)

# def biasvariable(shape):
#     init  =tf.random.normal(shape)
#     return tf.Variable(init)

# def maxpool(input):
#     return tf.nn.max_pool2d(input,[1,2,2,1],[1,2,2,1],"SAME")


# def Train():
#     tf.compat.v1.disable_eager_execution()
#     with tf.name_scope("data"):
#         x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,28,28,1])
#         y_true = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,10])

#     with tf.name_scope("model"):
#         with tf.name_scope("conv1"):
#             con_w_1 = weightvariable([3,3,1,10])
#             con_b_1 = biasvariable([10])
#             con_x_1 =tf.nn.relu( tf.nn.conv2d(x,con_w_1,[1,1,1,1],"SAME")+con_b_1)

#         with tf.name_scope("conv2"):
#             con_w_2 = weightvariable([3,3,10,1])
#             con_b_2  =biasvariable([1])
#             con_x_2 = tf.nn.relu(tf.nn.conv2d(con_x_1,con_w_2,[1,1,1,1],"SAME")+con_b_2)

#         with tf.name_scope("maxpool1"):
#             pool = maxpool(con_x_2)

#         with tf.name_scope("conv3"):
#             con_w_3 = weightvariable([3,3,1,1])
#             con_b_3 = biasvariable([1])
#             con_x_3 = tf.nn.relu(tf.nn.conv2d(pool,con_w_3,[1,1,1,1],"SAME")+con_b_3)

#         with tf.name_scope("maxpool2"):
#             pool2 = maxpool(con_x_3)
        
#         with tf.name_scope("fullConnectLayer"):
#             x_fc = tf.reshape(pool2,[-1,7*7])
#             with tf.name_scope("layer1"):
#                 w_fc_1 = weightvariable([49,16])
#                 b_fc_1 = biasvariable([16])
#                 x_fc_1 =tf.nn.relu( tf.matmul(x_fc,w_fc_1)+b_fc_1)
#             with tf.name_scope("outlayer"):
#                 w_fc_2 = weightvariable([16,10])
#                 b_fc_2 = biasvariable([10])
#                 y_pridict = tf.nn.softmax(tf.matmul(x_fc_1,w_fc_2)+b_fc_2)

#         with tf.name_scope("loss"):
#             loss = tf.reduce_mean(-y_true*tf.math.log(y_pridict))
#         with tf.name_scope("train"):
#             train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss)
#         init = tf.compat.v1.global_variables_initializer()

#         with tf.compat.v1.Session() as sess:
#             # dataset = getDataBases(data[0][0]/255,tf.one_hot(data[0][1],10))
#             sess.run(init)
#             for i in range(10000):
#                # d = tf.data.Dataset.batch(dataset,10)
#                 res= sess.run([train_step,con_x_1],{x:tf.cast(tf.reshape(data[0][0][i]/255,[1,28,28,1]),tf).eval(),y_true:tf.reshape(tf.one_hot(data[0][1][i],10),[1,10]).eval()})
#                 print(res)

            
# def test():
#     dataset = getDataBases(data[0][0]/255,tf.one_hot(data[0][1],10))
#     d = tf.data.Dataset.batch(dataset,10)

model = keras.Sequential(
        [
            keras.layers.Conv2D(32,3,activation=keras.activations.relu,padding="SAME"),
            keras.layers.Conv2D(16,3,activation=keras.activations.relu,padding="SAME"),
            keras.layers.AvgPool2D(strides=2,padding="SAME"),
            keras.layers.Conv2D(16,2,activation=keras.activations.relu,padding="SAME"),
            keras.layers.Conv2D(8,2,activation=keras.activations.relu,padding="SAME"),
            keras.layers.AvgPool2D(strides=2,padding="SAME"),
            keras.layers.Conv2D(8,2,activation=keras.activations.relu,padding="SAME"),
            keras.layers.AvgPool2D(strides=1,padding="SAME"),
            keras.layers.Flatten(),
            keras.layers.Dense(32,activation=keras.activations.relu),
            keras.layers.Dense(16,activation=keras.activations.relu),
            keras.layers.Dense(16,activation=keras.activations.relu),
            keras.layers.Dense(10,activation=keras.activations.softmax)
        ]
)

def Train():

    model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    data = keras.datasets.mnist.load_data()
    callback = keras.callbacks.TensorBoard(histogram_freq=10,update_freq=1,write_images=True)
    callback.model= model
    model.fit(tf.reshape(tf.cast( data[0][0]/255,tf.float32),[-1,28,28,1]),data[0][1],128,10)#,callbacks=[callback]
    model.save("./ckpt.h5")
def pridict(pridictimg):
    model.build([1,28,28,1])
    model.load_weights("./ckpt.h5")
    res=model.predict(pridictimg)
    print(res,tf.argmax(res,1).numpy())


'''
将[28,28,3]的图片处理成[1,28,28,1]的灰度图张量
'''
def img_precess(img):
    img_rel = []
    for i in img:
        temp = []
        for j in i:
            temp.append([(255-(j[0]+j[1]+j[2])/3.0)/255.0])

        img_rel.append(temp)
    imgrel=tf.reshape(tf.constant(img_rel),[1,28,28,1]).numpy()
    return imgrel
if __name__=="__main__":
    #Train()
    img= keras.preprocessing.image.load_img("./img/1 .jpg",target_size=(28,28))
    img = keras.preprocessing.image.img_to_array(img)
    # print(img_precess(img).shape)
    pridict(img_precess(img))



