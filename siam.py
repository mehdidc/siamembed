import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam, SGD
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255.0
left_input = Input((784,))
right_input = Input((784,))

model = Sequential([
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2),
])
encoded_l = model(left_input)
encoded_r = model(right_input)

def l1(tensor):
    return tf.reduce_sum(K.abs(tensor[0] - tensor[1]), axis=1, keepdims=True)
L1_layer = Lambda(l1)
pred = L1_layer([encoded_l, encoded_r])
siamese_net = Model(inputs=[left_input,right_input],outputs=pred)
optimizer = Adam(lr=0.001)
siamese_net.compile(loss="mean_squared_error",optimizer=optimizer)

X = x_train.copy().reshape((x_train.shape[0], 784))
Y = y_train.copy().astype(int)
Xl = X.copy()
Xr = X.copy()
np.random.shuffle(Xl)
np.random.shuffle(Xr)
dist = np.abs(Xl - Xr).sum(axis=1)
max_dist = dist.max()
med_dist = np.median(dist)
for i in range(300):
    np.random.shuffle(Xl)
    np.random.shuffle(Xr)
    dist = np.abs(Xl - Xr).sum(axis=1)
    dist[dist<med_dist]=0
    dist[dist>=med_dist]=5
    siamese_net.fit([Xl, Xr], dist, epochs=1, batch_size=512)
    if i % 10 == 0:
        H = model.predict(X)
        print(H.shape)
        fig = plt.figure(figsize=(50, 40))
        plt.scatter(H[:, 0], H[:, 1], c=Y, s=150,cmap = plt.cm.jet, marker='o')
        plt.savefig('out.png')
        plt.show()
