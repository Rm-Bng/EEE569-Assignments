"""
EEE569 spring 2024
Assignment 1 Part B
"""
from scipy.stats import multivariate_normal
from NNnodes import *


# 1 Generate non linearly sparable data 
CLASS1_SIZE = 100
CLASS2_SIZE = 100

MEAN1_1 = np.array([3, 10])
MEAN1_2 = np.array([10, 3])
COV1 = np.array([[1, 0], [0, 1]])

MEAN2_1 = np.array([3, 3])
MEAN2_2 = np.array([10, 10])
COV2 = np.array([[1, 0], [0, 1]])


X1_1 = multivariate_normal.rvs(MEAN1_1, COV1, int(CLASS1_SIZE/2))
X1_2 = multivariate_normal.rvs(MEAN1_2, COV1, int(CLASS1_SIZE/2))

X2_1 = multivariate_normal.rvs(MEAN2_1, COV2, int(CLASS2_SIZE/2))
X2_2 = multivariate_normal.rvs(MEAN2_2, COV2, int(CLASS2_SIZE/2))

X1 = np.vstack((X1_1, X1_2))
X2 = np.vstack((X2_1, X2_2))
X = np.vstack((X1, X2))

y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

TEST_SIZE = 0.25
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]


X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value =  t.value- learning_rate * t.gradients[t]
def accuracy(predictions, y_test):
        c = (predictions == y_test) # Elementwise comparison, the output vector [True,False, ...]
        correct = c.sum() # Number of True values
        accuracy = correct/len(predictions)
        return accuracy* 100
# Task 1 _________________________________________________
# Train using Logistic Regression
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
b = 0
A = np.random.randn(n_output,n_features) * 0.1

# Create nodes
x_node = Input()
y_node = Input()

b_node = Parameter(b)
A_node = Parameter(A)


# Build computation graph
l_node = Linear(x_node, A_node,b_node)
sigmoid = Sigmoid(l_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node, A_node, b_node,l_node,sigmoid,loss]
trainable = [A_node,b_node]

# Training loop
batch_size = 1
train_size = X_train.shape[0]
batches = train_size/batch_size


X_trainl = np.transpose(X_train)
y_trainl =  np.transpose(y_train)
epochs = 100
learning_rate = 0.01
for epoch in range(epochs): 
    for i in range(0,train_size,batch_size):
        x_node.value =  X_trainl[: , i: i+batch_size]
        y_node.value = y_trainl[i: i+batch_size]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

# Evaluate the model
X_testl = np.transpose(X_test)
correct_predictions = 0
for i in range(X_testl.shape[1]):
    x_node.value = X_testl[:,i]
    forward_pass(graph)

    prediction = ((sigmoid.value) >= 0.5).astype(int)
    if prediction == y_test[i]:
        correct_predictions += 1 

a = correct_predictions / X_testl.shape[1]
print(f"Accuracy using logistic regression: {a :.2f}%")
    
# Draw decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    x_node.value = np.array([i,j])
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

# Task 2 _________________________________________________
# Train using multilayer perceptron 
# Design MLP layers 
hidden = 2
width = 22
n_features = X_train.shape[1]

A1 = np.random.randn(width,n_features) 
A2 = np.random.randn(width,width) 
A3 = np.random.randn(1,width) 

b1 = np.zeros((width,1))
b2 = np.zeros((width,1))
b3 = 0

x_node = Input()
y_node = Input()

bn1 = Parameter(b1)
An1 = Parameter(A1)

bn2 = Parameter(b2)
An2 = Parameter(A2)

bn3 = Parameter(b3)
An3 = Parameter(A3)

hidden1_linear = Linear(x_node, An1, bn1)
hidden1_sigmoid = Sigmoid(hidden1_linear)

hidden2_linear = Linear(hidden1_sigmoid, An2,bn2)
hidden2_sigmoid = Sigmoid(hidden2_linear)

output_linear = Linear(hidden2_sigmoid, An3, bn3)
output_sigmoid = Sigmoid(output_linear)

loss = BCE(y_node, output_sigmoid)

graph = [An1,bn1,
         An2,bn2,
         An3,bn3,
         hidden1_linear, hidden1_sigmoid,
         hidden2_linear, hidden2_sigmoid,
         output_linear, output_sigmoid,
         loss]

trainable = [An1, An2, An3, bn1, bn2, bn3]

train_size = X_train.shape[0]
batch_size = 1
X_trainMLP = np.transpose(X_train)
y_trainMLP = np.transpose(y_train)
learning_rate = 0.1
for epoch in range(epochs):
    for i in range(0, train_size, batch_size):
        x_node.value = X_trainMLP[: , i: i+batch_size]
        y_node.value = y_trainMLP[i: i+batch_size]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)


# Evaluate the model
X_testMLP = np.transpose(X_test)
#correct_predictions = 0
predictions = []
for i in range(X_testMLP.shape[1]):
    x_node.value = X_testMLP[:,i].reshape(-1,1)
    forward_pass(graph)
    prediction = int(output_sigmoid.value.item() >= 0.5)
    predictions.append(prediction)


a = accuracy(predictions,y_test.astype(int))
print(f"Accuracy multilayer perceptron : {a:.2f}%")

Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([[i], [j]])  # Column vector input
    forward_pass(graph)
    Z.append(output_sigmoid.value.item())  # Ensure scalar

Z = np.array(Z).reshape(xx.shape)  # Match grid shape

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

# Task 3 _________________________________________________
hidden = 2
width = 22
n_features = X_train.shape[1]
n_output = len(np.unique(y))

learning_rate = 0.01
train_size = X_train.shape[0]
batch_size = 1

binary_classifier = MLP(n_features, hidden, width, n_output)
binary_classifier.train(X_train,y_train,batch_size,epochs,learning_rate)
binary_classifier.loss_epoch_curve()    

predictions = binary_classifier.predict(X_test)
a = accuracy(predictions,y_test)
print(f"MLP_v2 accuracy: {a :.2f}%")

#Draw decision boundary
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    binary_classifier.x_node.value = np.array([[i], [j]])
    z = binary_classifier.predict(np.array([[i,j]]))
    Z.append(z)

Z = np.array(Z).reshape(xx.shape)  # Match grid shape

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

# Task 4 _________________________________________________
from sklearn import datasets

mnist = datasets.load_digits()
X, y = mnist['data'],mnist['target'].astype(int)

# splitting data into 60% train and 40% test
TRAIN_SIZE = 0.6
train = int(len(X) * TRAIN_SIZE)

X_train, X_test = X[:train,:], X[train:,:]
y_train, y_test = y[:train], y[train:]


train_size = X_train.shape[0]
n_features = X_train.shape[1]
n_output = len(np.unique(y))

hidden = 1
width = 64

batch_size = 10
learning_rate = 0.05

multiClass_classifier = MLP(n_features, hidden, width, n_output)
multiClass_classifier.train(X_train,y_train,batch_size,100,learning_rate)
multiClass_classifier.loss_epoch_curve()    

predictions = multiClass_classifier.predict(X_test)
a = accuracy(predictions,y_test) 
print(f"MNIST classifier accuracy: {a :.2f}%")
