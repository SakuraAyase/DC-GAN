from keras.models import *
from keras.engine import *
from keras import *
import numpy as np
from keras.backend import *
from PIL import Image
from keras.layers import *
import matplotlib.pyplot as plt
from keras.optimizers import *
from tensorflow.examples.tutorials.mnist import input_data


def d_loss(y_true,y_pred):
    return mean(y_true*y_pred)


def plot(image,i):
    image = np.reshape(image, [28, 28])
    plt.imshow(image,cmap='gray')
    fileName = 'C:\\Users\\myfamily\\Desktop\\新建文件夹\\'
    plt.savefig(fileName+'lean'+str(i))

x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
x_train = x_train.reshape(-1, 28,\
        	28, 1).astype(np.float32)

D = Sequential()
dropout = 0.5
D.add(Conv2D(256,kernel_size=(2,2),strides=2,padding='same',input_shape=(28,28,1),activation='relu'))



D.add(Conv2D(64,kernel_size=(2,2),strides=1,padding='same',activation='relu'))


D.add(Conv2D(64,kernel_size=(2,2),padding='same',strides=1,activation='relu'))

D.add(Flatten())
D.add(Dense(1))
D.summary()

G = Sequential()

G.add(Dense(7*7*256,input_dim=100,activation='relu'))
G.add(Reshape((7,7,256)))



G.add(UpSampling2D())
G.add(Conv2DTranspose(128,(2,2),padding='same',activation='relu'))


G.add(UpSampling2D())
G.add(Conv2DTranspose(64,(2,2),padding='same',activation='relu'))



G.add(Conv2DTranspose(1,(2,2),padding='same',activation='sigmoid'))
G.summary()

DM = Sequential()
DM.add(D)
DM.compile(loss='mae',optimizer='RMSprop',metrics=['accuracy'])

AM = Sequential()
AM.add(G)
AM.add(D)
AM.compile(loss='mae',optimizer='RMSprop',metrics=['accuracy'])

GM = Sequential()
GM.add(G)
GM.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
noise = np.random.uniform(-1.0, 1.0, size=[256, 100])
images_train = x_train[:256]
GM.fit(noise,images_train,batch_size=128)
noise = np.random.uniform(-1.0, 1.0, size=[256, 100])
k = G.predict(noise)[0]

print('GM end')
plot(k, 0)


for i in range(20000):
    noise = np.random.uniform(-1.0, 1.0, size=[256, 100])
    img_faker = G.predict(noise)

    images_train = x_train[np.random.randint(0,x_train.shape[0], size=256), :, :, :]
    x = np.concatenate((images_train,img_faker))
    y = np.zeros((512,1))
    y[:256,:] = 1
    print('DM')

    DM.fit(x, y, epochs=1, batch_size=128)
    for I in D.layers:
        weight = I.get_weights()
        weight = [np.clip(w, -0.01, 0.01) for w in weight]
        I.set_weights(weight)
    print(DM.predict(x)[0])
    print(DM.predict(x)[511])

    y = np.ones((256,1))

    print('AM')
    AM.fit(noise,y,epochs=1,batch_size=128)
    noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
    k = G.predict(noise)[0]
    k.shape = (28,28)

    plot(k,i+1)
    print(i)





