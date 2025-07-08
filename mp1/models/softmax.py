import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        batch_size = X_train.shape[0]
        num_features = X_train.shape[1]

        # Initialize the weights if not done so already
        if self.w is None:
            self.w = np.random.randn(num_features, self.n_class)

        # Calculate the score matrix
        score_matrix = X_train.dot(self.w)
        
        # Normalize the score matrix to prevent overflow
        
        score_matrix = (score_matrix - np.max(score_matrix, axis=1).reshape(batch_size, 1))
        # Calculate the probability matrix
        
        prob_matrix = np.exp(score_matrix) / np.sum(np.exp(score_matrix), axis=1, keepdims=True).reshape(batch_size, 1)
        
        # Initialize the gradient matrix
        grad_matrix = np.zeros((num_features, self.n_class))

        for i in range(batch_size):
            for class_ in range(self.n_class):
                if class_ == y_train[i]:
                    grad_matrix[:, class_] += self.reg_const*self.w[:,class_]/self.n_samples+(prob_matrix[i, class_] - 1) * X_train[i]
                else:
                    grad_matrix[:, class_] += self.reg_const*self.w[:,class_]/self.n_samples+prob_matrix[i, class_] * X_train[i]

        grad_matrix /= batch_size
        return grad_matrix

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_samples = X_train.shape[0]
        batch_size=256
        n_samples, n_features = X_train.shape
        b = np.ones((n_samples,1))
        X_train=np.hstack((X_train,b))
        self.n_samples= len(X_train)
        for epoch in range(self.epochs):
          
          I_permutation= np.random.permutation(n_samples)
          X_train =  X_train[I_permutation,:]
          y_train =  y_train[I_permutation]  
          for i in range(0, n_samples, batch_size):
              X_batch = X_train[i:i + batch_size,:]
              y_batch = y_train[i:i + batch_size]

              grad = self.calc_gradient(X_batch, y_batch)
              self.w -= self.lr * grad
          if epoch!=0 and epoch%5==0:
            self.lr=self.lr*0.01      
         
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        N, D = X_test.shape
        b = np.ones((N,1))
        X_test=np.hstack((X_test,b))
        score_matrix = X_test.dot(self.w)
        predictions = np.argmax(score_matrix, axis=1)

        return predictions