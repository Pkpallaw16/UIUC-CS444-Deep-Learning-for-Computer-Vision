"""Logistic regression model."""

import numpy as np

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None # TODO: change this 
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-1*z))


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        self.w = np.zeros((X_train.shape[1]+1))
        X_train = np.hstack((X_train,np.ones((X_train.shape[0], 1))))
        y = y_train.copy()
        y[y == 0] = -1
      
        for epochs in range(self.epochs): 
          # Update w values as a gradient of the cost function
          for sample in range(X_train.shape[0]):
            z = np.dot(X_train[sample,:],self.w)
            self.w += self.lr*self.sigmoid(-1*np.dot(y[sample],z))*y[sample]*X_train[sample,:]
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.hstack((X_test,np.ones((X_test.shape[0], 1))))
        z = np.dot(X_test,self.w.T)
        prob = self.sigmoid(z)
        
        return [1 if prob[i] >= self.threshold  else 0 for i in range(len(prob))]
