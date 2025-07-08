"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        fc=np.matmul(self.w,X_train.T).T
        batch_size, n_features = X_train.shape
        gradientes= np.zeros([self.n_class, n_features])
        loss=[]
        #grad = np.zeros((self.n_class, X_train.shape[1]))
        for row_x, row_fc,true_class in zip(X_train,fc,y_train):
            gradiente_sample= np.zeros([self.n_class, n_features])
            loss_row= 0
            for class_ in range(self.n_class):
              loss_row += max(0,1-row_fc[true_class]+row_fc[class_])
              if class_ == true_class: 
                  continue
              margin = (row_fc[true_class]-row_fc[class_])
              if margin < 1:
                  gradiente_sample[true_class,:] += row_x
                  gradiente_sample[class_,:] = row_x
                  #print(grad) 
            gradiente_sample[true_class,:]= self.reg_const*self.w[true_class,:]/self.n_samples -gradiente_sample[true_class,:]
            # gradiente_sample[true_class,:]= self.reg_const*self.w[true_class,:] -gradiente_sample[true_class,:]
            non_class= list(np.arange(0,self.n_class)).remove(true_class)
            gradiente_sample[non_class,:]= self.reg_const*self.w[non_class,:]/self.n_samples + gradiente_sample[non_class,:]
            # gradiente_sample[non_class,:]= self.reg_const*self.w[non_class,:] + gradiente_sample[non_class,:]

            gradientes += gradiente_sample
            loss.append(loss_row)
        
        return  (gradientes/batch_size, np.mean(loss))          

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        epoch_loss=[]
        batch_size=256
        self.n_samples= len(X_train)
        n_samples, n_features = X_train.shape
        b = np.ones((n_samples,1))
        X_train=np.hstack((X_train,b))
        self.w = np.random.rand(self.n_class, n_features+1)
        for epoch in range(self.epochs):
          if epoch!=0 and epoch%5==0:
            self.lr=self.lr*0.1
          I_permutation= np.random.permutation(n_samples)
          X =  X_train[I_permutation,:]
          Y =  y_train[I_permutation]
          for i in range(0,n_samples,batch_size):
                  X_batch= X[i:batch_size+i,:]
                  Y_batch= Y[i:batch_size+i]
                  gradientes,loss= self.calc_gradient(X_batch, Y_batch)  
                  self.w= self.w - self.lr*gradientes  
          epoch_loss.append(loss)
          #pred_svm = svm_fashion.predict(X_test_fashion)
          #print('The testing accuracy is given by: %f' % (get_acc(pred_svm, y_test_fashion)))
          #print (f'Error in Epoch {epoch} is {np.mean(epoch_loss)}')
            #self.lr=self.lr*0.1
            #self.w =  grad
            
        return

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
        N, D = X_test.shape
        b = np.ones((N,1))
        X_test=np.hstack((X_test,b))
        y_pred = np.argmax(np.dot(X_test,self.w.T) , axis=1)
        return y_pred
        
