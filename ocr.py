from time import time

import cv2
import numpy
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Dense
from keras.models import Sequential

import numpy as np

start = time()

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11'] / m['mu02']
    # Calculate affine transform to correct skewness.
    SZ = 28
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


(X_mnist, y_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
red, kolona = X_mnist.shape[1:]

for i in range(len(X_mnist)):
    X_mnist[i] = deskew(X_mnist[i])

d = X_mnist.copy()

X_mnist = X_mnist.reshape(X_mnist.shape[0], red, kolona, 1)
X_test_mnist = X_test_mnist.reshape(X_test_mnist.shape[0], red, kolona, 1)

y = numpy.zeros((len(y_mnist), 10), numpy.int)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(red, kolona, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

for i in range(len(y)):
    y[i][y_mnist[i]] = 1

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_mnist, y, epochs=30, batch_size=5, verbose=1)

print("Training completed in: " + str(time() - start))

json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json)

model.save_weights("model.h5")

print(print(np.sum(model.predict(X_test_mnist) == y_test_mnist) / y_test_mnist.shape))
