"""Neural network model."""

from collections import defaultdict
from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str, 
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        self.t = 0

        self.m = defaultdict(int)
        self.v = defaultdict(int)

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, self.num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        
            

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        output = np.dot(X,W) + b
        return output

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        
        output = np.maximum(0,X)
        return output

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
    
        X = np.where(X >= 0, 1, 0)
        return X
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
  
    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
		# compute the derivative of the sigmoid function
      return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
      # n_samples, n_outputs = y.shape
      # mse_per_output = np.sum((y - p)**2, axis=0) / n_samples
      # mse_total = np.mean(mse_per_output)
      # return mse_total
      return np.mean((y - p)**2)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs = {}
        self.outputs['A0'] = X.copy()
        
        for l in range(1, self.num_layers):
          self.outputs["Z" + str(l)] = self.linear(self.params["W" + str(l)], self.outputs["A" + str(l-1)], self.params["b" + str(l)])
          self.outputs["A" + str(l)] = self.relu(self.outputs["Z" + str(l)])
          
        # Pass output of second last layer through last layer 
        self.outputs["Z" + str(self.num_layers)] = self.linear(self.params["W" + str(self.num_layers)], self.outputs["A" + str(self.num_layers-1)], self.params["b" + str(self.num_layers)])
        self.outputs["A" + str(self.num_layers)] = self.sigmoid(self.outputs["Z" + str(self.num_layers)])
        
        return self.outputs["A" + str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        num_samples = (y.shape[0]*y.shape[1])
        
        #Compute loss 
        loss = self.mse(self.outputs["A" + str(self.num_layers)], y)
        
        #upstream gradient of last layer 
        self.gradients['A' + str(self.num_layers)] = 2 * (self.outputs["A" + str(self.num_layers)] - y) / num_samples
        
        # Downstream gradient of last layer 
        self.gradients['Z' + str(self.num_layers)] = self.gradients['A' + str(self.num_layers)] * self.sigmoid_grad(self.outputs["Z" + str(self.num_layers)])        
        self.gradients["W" + str(self.num_layers)] = np.dot(self.outputs["A" + str(self.num_layers - 1)].T, self.gradients['Z' + str(self.num_layers)])   
        self.gradients["b" + str(self.num_layers)] = np.sum(self.gradients['Z'+ str(self.num_layers)], axis=0)
        
        dAPrev = np.dot(self.gradients['Z' + str(self.num_layers)], self.params["W" + str(self.num_layers)].T)
       
        
        for l in range(self.num_layers - 1, 0, -1):
          self.gradients["Z" + str(l)] = dAPrev * self.relu_grad(self.outputs["Z" + str(l)])
          self.gradients["W" + str(l)] = np.dot(self.outputs["A" + str(l - 1)].T,self.gradients["Z" + str(l)])
          self.gradients["b" + str(l)] = np.sum(self.gradients["Z" + str(l)], axis = 0)
          if l > 1:
            dAPrev = self.gradients["Z" + str(l)].dot(self.params["W" + str(l)].T)
        return loss 

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if self.opt == "SGD":
          for l in range(1, self.num_layers + 1):
              self.params['W'+str(l)] += -lr * self.gradients['W'+str(l)] 
              self.params['b'+str(l)] += -lr * self.gradients['b'+str(l)] 
              
        elif self.opt == "adam":
          self.t += 1 
          for l in range(1, self.num_layers+1 ):
            self.m['W'+str(l)] = b1*self.m['W'+str(l)] + (1-b1)*self.gradients['W'+str(l)] 
            self.m['b'+str(l)] = b1*self.m['b'+str(l)] + (1-b1)*self.gradients['b'+str(l)] 
            self.v['W'+str(l)] = b2*self.v['W'+str(l)] + (1-b2)*(self.gradients['W'+str(l)]**2)
            self.v['b'+str(l)] = b2*self.v['b'+str(l)] + (1-b2)*(self.gradients['b'+str(l)]**2)

            m_dw_corr = self.m['W'+str(l)]/(1-b1**self.t)
            m_db_corr = self.m['b'+str(l)]/(1-b1**self.t)
            v_dw_corr = self.v['W'+str(l)]/(1-b2**self.t)
            v_db_corr = self.v['b'+str(l)]/(1-b2**self.t)

          ## update weights and biases
            self.params['W'+str(l)] -=  lr*(m_dw_corr/(np.sqrt(v_dw_corr)+eps))
            self.params['b'+str(l)] -=  lr*(m_db_corr/(np.sqrt(v_db_corr)+eps)) 
        
  
        #pass