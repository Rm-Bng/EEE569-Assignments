import numpy as np
from matplotlib import pyplot
from keras.datasets import mnist
from cnnv2 import *


# loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

train_X = train_X/255.
test_X = test_X/255.


train_size, _, _, ci = train_X.shape


classifier = CNN([convolution(input_channels=ci, kernal_HW=3, number_of_kernels=16),
                  LeakyReLU(),
                  max_pooling(pool_size=2, stride=2),

                  convolution(input_channels=16, kernal_HW=2,
                              number_of_kernels=32),
                  LeakyReLU(),
                  max_pooling(pool_size=2, stride=2),

                  convolution(input_channels=32, kernal_HW=2,
                              number_of_kernels=64),
                  LeakyReLU(),
                  max_pooling(pool_size=2, stride=2),

                  convolution(input_channels=64, kernal_HW=2,
                              number_of_kernels=128),
                  LeakyReLU(),
                  flatten(),

                  Dense(input_channels=128, width=64),
                  LeakyReLU(),

                  Dense(input_channels=64, width=10),

                  SoftMax(),
                  CE()
                  ])

learning_rate = 0.0001
batch_size = 32
epochs = 1
classifier.train(train_X, train_y, batch_size, epochs, learning_rate)

predictions = classifier.predict(test_X)
a = accuracy(predictions, test_y)
print(f"Accuracy: {a :.2f}%")
