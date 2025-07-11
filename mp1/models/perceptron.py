"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        batch_size=1000
        N, D = X_train.shape
        b = np.ones((N,1))
        X_train=np.hstack((X_train,b))
        self.w = np.random.rand(self.n_class,X_train.shape[1])
        for epoch in range(self.epochs):
          if epoch!=0 and epoch%5==0 and self.lr>0.00001:
            self.lr=self.lr*0.1
          epoch_loss=[]
          I_permutation=np.random.permutation(N)
          X=X_train[I_permutation,:]
          Y=y_train[I_permutation]
          for i in range(0, N,batch_size):
              x_batch=X[i:batch_size+i,:]
              y_batch=Y[i:batch_size+i]
              y_batch=y_batch
              gradients=np.zeros([self.n_class,X_train.shape[1]])

              for x,y in zip(x_batch,y_batch):
                
                    # calculate prediction
                prediction = np.matmul(self.w,x)
                
                loss=0
                for class_c in range(0,self.n_class):
                  if class_c==y:
                    continue
                  loss+=max(0,prediction[class_c]-prediction[y])
                epoch_loss.append(loss)    
                response_class=prediction[y]
                for clas in range(self.n_class):
                  if prediction[clas]>response_class:
                    gradients[clas,:]-=x
                    gradients[y,:]+=x
                  
                self.w =self.w+ self.lr*gradients/batch_size
                
                
      

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
        N, D = X_test.shape
        b = np.ones((N,1))
        X_test=np.hstack((X_test,b))
        predictions = np.dot(self.w,X_test.T) 
        return np.argmax(predictions, axis=0)
        
