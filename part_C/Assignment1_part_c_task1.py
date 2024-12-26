from NeuralNetwork import *
import numpy as np
from matplotlib import pyplot
from keras.datasets import mnist


def accuracy(predictions, y_test):
        c = (predictions == y_test) # Elementwise comparison, the output vector [True,False, ...]
        correct = c.sum() # Number of True values
        accuracy = correct/len(predictions)
        return accuracy* 100



#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

train_X = train_X/255.
test_X = test_X/255.


train_size = train_X.shape[0]
train_X = train_X.reshape(train_size, -1)
test_size = test_X.shape[0]
test_X = test_X.reshape(test_size, -1)

n_features = train_X.shape[1]
n_output = len(np.unique(train_y))


# Model 1

learning_rate = 0.0001
batch_size = 8
epochs = 50

classifier = NN([Linear_mlp( n_features, 128),
                 ReLU(),
                 Linear_mlp(128, 10),
                 SoftMax(),
                 CE()
                    ]) 

classifier.train(train_X,train_y,batch_size,epochs,learning_rate)
predictions = classifier.predict(test_X)
a = accuracy(predictions,test_y)
print(f"Model 1 accuracy: {a :.2f}%")
classifier.loss_epoch_curve()    

# Model 4
learning_rate = 0.0001
batch_size = 32
epochs = 20

classifier = NN([Linear_mlp( n_features, 512),
                 ReLU(),
                 Linear_mlp( 512, 256),
                 ReLU(),
                 Linear_mlp( 256, 128),
                 ReLU(),
                 Linear_mlp( 128, 64),
                 ReLU(),
                 Linear_mlp(64, 10),
                 SoftMax(),
                 CE()
                    ]) 


classifier.train(train_X,train_y,batch_size,epochs,learning_rate)
predictions = classifier.predict(test_X)
a = accuracy(predictions,test_y)
print(f"Model 4 accuracy: {a :.2f}%")
classifier.loss_epoch_curve()    

# model 5
learning_rate = 0.01
batch_size = 32
epochs = 50

classifier = NN([Linear_mlp( n_features, 128),
                 ReLU(),
                 Linear_mlp( 128, 64),
                 ReLU(),
                 Linear_mlp( 64, 16),
                 ReLU(),
                 Linear_mlp(16, 10),
                 SoftMax(),
                 CE()
                    ]) 


classifier.train(train_X,train_y,batch_size,epochs,learning_rate)
predictions = classifier.predict(test_X)
a = accuracy(predictions,test_y)
print(f"Model 5 accuracy: {a :.2f}%")
classifier.loss_epoch_curve() 

