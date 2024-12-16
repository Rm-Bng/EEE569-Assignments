"""
EEE569 Spring 2024
Assignment 1 Part A
"""

from NNnodes import *
from scipy.stats import multivariate_normal
import time

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
#EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([0, -1])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([-1, 2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)
# Combine the points and generate labels
X = np.vstack((X1, X2))

y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
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

'''
Training

1) Parameters
A: Vector of initial weights
b: Initial bias (scalar or vector)
X_train: Matrix of training samples
y_train: Vector of training sample targets
epochs: Number of iterations over the entire dataset
learning_rate: Step size used to update weights with respect to gradients
batch_size: Number of samples used in one forward and backward pass during the learning process

2) Variables
train_size: Total number of training samples
batches: Number of batches per epoch
loss_value: Average loss for each epoch
losses: List storing the average loss for use in plots
start, end, elapsed_time: Time variables used to calculate the total training time

3) How it works
Each epoch is divided into batches. The inner loop processes a batch of batch_size training examples to update the weights. The average loss is computed after each batch and accumulated.

'''
def Training(X_train, y_train, epochs, learning_rate, batch_size):
    train_size = X_train.shape[0]
    batches = train_size/batch_size
    A_node.value = np.random.randn(n_output,n_features) * 0.1
    b_node.value = 0
    X_train = np.transpose(X_train)
    y_train =  np.transpose(y_train)
    
    losses = []
    
    start = time.time()
    for epoch in range(epochs):
        loss_value = 0
        for i in range(0,train_size,batch_size):
            x_node.value =  X_train[: , i: i+batch_size]
            y_node.value = y_train[i: i+batch_size]
            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, learning_rate)
        
            loss_value += loss.value
        losses.append(loss_value/batches)
        
    end = time.time()
    elabsed_time = end - start
    return losses , elabsed_time  


def batch_size_impact(X_train,y_train,epochs,batch_range,learning_rate):
    x = [i for i in range(1, epochs+1)]
    
    elabsed_time = []
    batches = [i for i in range (*batch_range)]

    for i in (batches):
        losses, tm = Training(X_train,y_train,epochs,learning_rate,i)
        elabsed_time.append(tm)
        plt.plot(x, losses,label='batch size= '+str(i))

    plt.title('effect of batch size on loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.bar(batches, elabsed_time)

    # Add title and labels
    plt.title('effect of batch size on time ')
    plt.xlabel('Batch size ')
    plt.ylabel('Time')

    # Show the plot
    plt.show()

# Model parameters
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

epochs = 100
learning_rate=0.01

# Training loop
batch_size = 1
train_size = X_train.shape[0]
batches = train_size/batch_size

X_trainT = np.transpose(X_train)
y_trainT =  np.transpose(y_train)
epochs = 100
learning_rate = 0.001
for epoch in range(epochs): #loops
    for i in range(0,train_size,batch_size):
        loss_value = 0
        x_node.value =  X_trainT[: , i: i+batch_size]
        y_node.value = y_trainT[i: i+batch_size]
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)
        
    
        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss.value}")


#evaluate model
correct_predictions = 0
X_test = np.transpose(X_test)

for i in range(X_test.shape[1]):
    x_node.value = X_test[:,i].reshape(2,1)
    forward_pass(graph)
    
    prediction = ((sigmoid.value) >= 0.5).astype(int)
    if prediction == y_test[i]:
        correct_predictions += 1 
    
accuracy = correct_predictions / X_test.shape[1]
print(f"Accuracy: {accuracy * 100:.2f}%")


#Decision boundary
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

learning_rate=0.1

#impact of batch size

batch_range = (1,11)
batch_size_impact(X_train,y_train,epochs,batch_range,learning_rate)

batch_range = (2,13,2)
batch_size_impact(X_train,y_train,epochs,batch_range,learning_rate)


batch_range = (10,151,20)
batch_size_impact(X_train,y_train,epochs,batch_range,learning_rate)


#learning rate impact
x = [i for i in range(1, epochs+1)]

rates = [0.00001,0.0001,0.001,0.01,1,2]
for i in (rates):
    losses,t = Training(X_train,y_train,epochs,i,1)
    plt.plot(x, losses,label='learning rate= '+str(i))

plt.title('effect of learning rate on loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

