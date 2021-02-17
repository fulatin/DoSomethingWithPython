import PIL.Image
import tensorflow as tf
import cv2
import tensorflow.keras.layers
import tensorflow.keras.losses
import tensorflow.keras.optimizers
from tensorflow import keras


model = keras.Sequential(
        [
            keras.layers.Conv2D(64,3,activation=keras.activations.relu,padding="SAME",kernel_regularizer=keras.regularizers.l1()),
            keras.layers.Conv2D(56,3,activation=keras.activations.relu,padding="SAME"),
            keras.layers.MaxPool2D(strides=2,padding="SAME"),
            keras.layers.Conv2D(28,3,activation=keras.activations.relu,padding="SAME"),
            keras.layers.Conv2D(28,3,activation=keras.activations.relu,padding="SAME"),
            keras.layers.MaxPool2D(strides=2,padding="SAME"),
            keras.layers.Flatten(),
            keras.layers.Dense(64,activation=keras.activations.relu,kernel_regularizer=keras.regularizers.l1()),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32,activation=keras.activations.relu),
            keras.layers.Dense(10,activation=keras.activations.softmax)
        ]
)

def Train():
    model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    data = keras.datasets.mnist.load_data()
    callback = keras.callbacks.TensorBoard(histogram_freq=100,update_freq=1,write_images=True)
    callback.model= model
    model.fit(tf.reshape(tf.cast( data[0][0]/255,tf.float32),[-1,28,28,1]),data[0][1],64,4)#,callbacks=[callback]
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
    # Train()
    img= keras.preprocessing.image.load_img("./img/1.jpg",target_size=(28,28))
    img = keras.preprocessing.image.img_to_array(img)
    pridict(img_precess(img))


 
