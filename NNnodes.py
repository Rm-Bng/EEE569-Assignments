"""
2024
EE569 Home work 1 part (a-b)
Reem Ben Guma 
"""

'''
This module contains a class implementation for running simple NN architecture
class:
    - Node: Base Node 
    - Input: For features and labels
    - Parameter: For weights and biases
    - Linear: Represent node which performs (weights*features+bias) operation
    - Linearv2: Second version of Linear where parameter creation and initialization performed within the node
    - Sigmoid: Activation function its output between [0,1]
    - Softmax: Activation function used in the last layer for multiclass classification
    - BCE: Binary Cross Entropy the loss function for binary classification
    - CE: Cross Entropy the loss function for multiclass classification
    - MLP: Multi Layer Perceptron used to reduce the need of manual coding
'''
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] =self.gradients[self]+ n.gradients[self]


class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] = n.gradients[self]

class Linear(Node):
    def __init__(self, X, A, b):
        Node.__init__(self, [A,X,b])


    def forward(self):
        '''
        linear = AX + b
        X-> size( input_features x batch_size)
        A-> size(layerWidth x input_features)
        b-> size(layerWidth x 1)
        '''
        A,X, b = self.inputs
        self.value = np.matmul(A.value,X.value) + b.value

    def backward(self):
        # The gradient of linear with respect to its inputs
        A, X, b = self.inputs
        
        batch_size = X.value.shape[1]
        '''
        linear = AX + b
        dloss/dX = dloss/dlinear * dlinear/dX = dloss/dlinear * A
        dloss/dA = dloss/dlinear * dlinear/dA = dloss/dlinear * X
        dloss/dA = dloss/dlinear * dlinear/db = dloss/dlinear * 1
        
        where dloss/dlinear-> size(1 x batch_size)
        
        with multi batch we take the average of gradients
        '''
        self.gradients[A] = (1/batch_size) *  np.matmul(self.outputs[0].gradients[self],np.transpose(X.value) ) 
        self.gradients[X] = np.matmul(np.transpose(A.value), self.outputs[0].gradients[self])
        self.gradients[b] =(1/batch_size)* np.sum(self.outputs[0].gradients[self], axis=1,keepdims=True) 


class Linear_mlp(Node):
    def __init__(self,X,input_channels,width):
        Node.__init__(self, [X])
        self.width = width
        self.input_channels = input_channels
        self.parameters_initialization()
    
    def parameters_initialization(self):
        '''
        A-> size(layerWidth x input_features)
        b-> size(layerWidth x 1)
        
        Xavier Weight Initialization
        The xavier initialization method is calculated as a random number
        with a uniform probability distribution between the range -(1/sqrt(n)) and 1/sqrt(n),
        where n is the number of inputs to the node.
        '''
        self.parameters= []
        xv = np.sqrt(1.0/self.input_channels) 
        A = np.random.uniform(-xv,xv,(self.width,self.input_channels))
        #A = np.random.randn(self.width,self.input_channels)
        b = np.zeros((self.width,1))
        
        self.parameters.append(Parameter(A))
        self.parameters.append(Parameter(b))
        
        '''
        The process of adding linear as an output node of parameters was done when parameters sent 
        as an input to linear, now parameters are created within the linear node 
        '''
        self.parameters[0].outputs.append(self)
        self.parameters[1].outputs.append(self)
            
    def forward(self):
        '''
        linear = AX + b
        linear-> size(layerWidth x batch_size)
        
        X-> size(input_features x batch_size)
        A-> size(layerWidth x input_features)
        b-> size(layerWidth x 1)
        '''
        X = self.inputs[0]
        A,b = self.parameters
        self.value = np.matmul(A.value,X.value) + b.value

    def backward(self):
        '''
        linear = AX + b
        dloss/dX = dloss/dlinear * dlinear/dX = dloss/dlinear * A
        dloss/dA = dloss/dlinear * dlinear/dA = dloss/dlinear * X
        dloss/db = dloss/dlinear * dlinear/db = dloss/dlinear * 1
        
        where dloss/dlinear-> size(width x batch_size)
        
        with multi batch we take the average of gradients
        '''
        X = self.inputs[0]
        A,b = self.parameters
        
        batch_size = X.value.shape[1]

        self.gradients[A] = (1/batch_size) *  np.matmul(self.outputs[0].gradients[self],np.transpose(X.value) ) 
        self.gradients[X] = np.matmul(np.transpose(A.value), self.outputs[0].gradients[self])
        self.gradients[b] =(1/batch_size)* np.sum(self.outputs[0].gradients[self], axis=1,keepdims=True) 
        A.backward()
        b.backward()

# Activation Functions
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        input_value =self.inputs[0].value

        #clipping to avoid overflow
        self.value = np.clip(1 / (1 + np.exp(-input_value)), 1e-12, 1-1e-12)
        #self.value = 1 / (1 + np.exp(-input_value))

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = self.outputs[0].gradients[self]*partial

class SoftMax(Node):
    def __init__(self, X):
        Node.__init__(self, [X])
        
    def forward(self):
        '''
        SoftMax = exp(input[i])/ sum (exp(input[j])) where j = 0,1,... ,number_of_classes-1
        SoftMax-> size(number_of_classes x batch_size)
        
        X-> size(input_features x batch_size)
        A-> size(number_of_classes x input_features)
        b-> size(number_of_classes x 1)
        '''
        input_value = self.inputs[0].value
        unNormalized = np.exp(input_value)
        self.value =  unNormalized / np.sum(unNormalized, axis=0, keepdims=True)
    
    def backward(self):
        X = self.inputs[0]
        '''
        ∂Si/∂xj = ∂/∂xj(exp(xi)/sum(exp(xk))) where k = 0,1,... ,number_of_classes-1
        
        ∂Si/∂xj = {Si(1-Sj)  if i=j,
                  -SiSj     if i≠j}
        
        ∂Si/∂xj = Si(δij−Sj)
        
        δij = {1  if i=j,
               0  if i≠j}        
        '''
        gd1 = self.value * self.outputs[0].gradients[self]
        gd1 = np.sum(gd1, axis=0)
        self.gradients[X] = (self.value*(self.outputs[0].gradients[self] - gd1))
        

# Loss Functions
class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-8 #to avoid overflow
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.value = np.sum(-y_true.value*np.log(y_pred_clipped)-(1-y_true.value)*np.log(1-y_pred_clipped))
        self.value = self.value/ y_true.value.shape[0]
        
    def backward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-8
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.gradients[y_pred] =  (1 / y_true.value.shape[0]) * np.sum((y_pred_clipped - y_true.value)/(y_pred_clipped*(1-y_pred_clipped)))
        # gradient with respect to actual class labels not needed
        #self.gradients[y_true] = (np.log(y_pred.value) - np.log(1-y_pred.value))
        
class CE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        '''
        Cross Entropy Loss = sum(y_true[i] * ln(y_pred[i])) where i = 0,1,... ,number_of_classes-1
        CE-> size(1 x batch_size)
        
        y_true-> size(number_of_classes x batch_size)
        y_pred-> size(number_of_classes x batch_size)
        '''
        y_true, y_pred = self.inputs
        # to avoid overflow
        epsilon = 1e-8 
        y_pred_safe = np.clip(y_pred.value, epsilon, 1 - epsilon)  
        self.value = -np.sum(y_true.value * np.log(y_pred_safe))
        # average loss
        self.value = (1/y_true.value.shape[1]) * self.value

    def backward(self):
        '''
        dloss/dsoftmax_input = -yi/softmax(xi)
        '''
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = -y_true.value/(y_pred.value)


class MLP:
    def __init__(self, input_channels, hidden_layers, width, n_class):
        self. input_channels = input_channels #input features
        self.hidden_layers = hidden_layers
        self.width = width
        self.n_class = n_class #number of classes in the dataset
        self.__network_creation() #crate network architecture

    def __network_creation(self):

        #layers -> list contains the width of each layer
        layers = [self.width for i in range(self.hidden_layers)]
        layers.insert(0, self.input_channels)
        layers.append(self.n_class)

        self.x_node = Input()
        self.y_node = Input()

        self.graph = []
        self.trainable = []

        prev = self.x_node
        
        for i in range(self.hidden_layers):
            hidden = Linear_mlp(prev,layers[i],layers[i+1]) #send node's width and number of inputs to the linear node
            sigmoid = Sigmoid(hidden)
            prev = sigmoid
            self.graph += [hidden,sigmoid]
            self.trainable.extend(hidden.parameters)
            
        # to design output layer based on the type of problem (binary-multiclass)
        if (self.n_class == 2):
            output_linear = Linear_mlp(prev,self.width,1)          
            output_activation = Sigmoid(output_linear)
            loss = BCE(self.y_node, output_activation)
            
        else:
            output_linear = Linear_mlp(prev,self.width,self.n_class)          
            output_activation = SoftMax(output_linear)
            loss = CE(self.y_node, output_activation)
        
        self.graph += [output_linear,output_activation,loss]
        self.trainable.extend(output_linear.parameters)        
        
        
        
    def train(self,X,y,batch_size,epochs,learning_rate=1e-2):
        self.losses=[]
        '''
        MLP used to train data with different output dimensions
            -Binary classification: y->size(1 x batch_size)
            -Multiclass classification: y->size(number_of_classes x batch_size)
        
        using if statement each time it dealing with y is inefficient so two versions of training are used
        
        '''
        if (self.n_class > 2):
            self.__multiclass_train(X,y,batch_size,epochs,learning_rate)
        else:
            self.__binary_train(X,y,batch_size,epochs,learning_rate)

    
    def __multiclass_train(self,X,y,batch_size,epochs,learning_rate): 
        X = X.T # to be in the shape(input_features x batch_size)
        train_size = X.shape[1]
        betches = train_size/batch_size
        y = self.__onehot_encoded(y)

        for epoch in range(epochs):
            ep_loss = 0
            for i in range(0, train_size, batch_size):
                self.x_node.value = X[: , i: i+batch_size]
                self.y_node.value = y[: , i: i+batch_size]

                self.__forward_pass()
                self.__backward_pass()
                self.__sgd_update(learning_rate)
                ep_loss += self.graph[-1].value 
            self.losses.append(ep_loss/betches)

    def __binary_train(self,X,y,batch_size,epochs,learning_rate):
        X = X.T # to be in the shape(input_features x batch_size)
        train_size = X.shape[1]
        betches = train_size/batch_size

        for epoch in range(epochs):
            ep_loss = 0
            for i in range(0, train_size, batch_size):
                self.x_node.value = X[: , i: i+batch_size]
                self.y_node.value = y[i: i+batch_size]

                self.__forward_pass()
                self.__backward_pass()
                self.__sgd_update(learning_rate)
                ep_loss += self.graph[-1].value 
            self.losses.append(ep_loss/betches)
        
            
            
    def __forward_pass(self):
        for n in self.graph:
            n.forward()

    def __backward_pass(self):
        for n in self.graph[::-1]:
            n.backward()

        # SGD Update
    def __sgd_update(self,learning_rate):
        for t in self.trainable:
            t.value =  t.value- learning_rate * t.gradients[t]
            
    def __onehot_encoded(self,y):
        number_of_classes = self.n_class
        yh = np.zeros((number_of_classes, y.shape[0]))
        
        '''enumerate is a Python function that returns both
          the index and the value of each element in y'''
        for idx, value in enumerate(y.astype(int)):
            yh[value, idx] = 1
        
        return yh
                
    def predict(self, X):
        # return list of the predicted labels
        if (self.n_class == 2):
            y = self.__binary_predict(X)
        else:
            y = self.__multiclass_predict(X)
        return y

    def __binary_predict(self,X):
        X = X.T
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            self.x_node.value = X[:,i].reshape(-1,1)
            self.__forward_pass()
            y[i] = round(float(self.graph[-2].value.item()))
        return y
    
    def __multiclass_predict(self,X):
        X = X.T
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            self.x_node.value = X[:,i].reshape(-1,1)
            self.__forward_pass()
            y[i] = np.argmax(self.graph[-2].value, axis=0)
        return y            
      
    def loss_epoch_curve(self):
        epochs = [i+1 for i in range(len(self.losses))]
        plt.plot(epochs, self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss vs epoch curve')
        plt.show()
