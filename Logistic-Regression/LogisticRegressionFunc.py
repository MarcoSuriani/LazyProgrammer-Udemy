# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:48:05 2019

@author: Suriani
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionSuriani:
    def fit(self, x, y, lrate=0.001, nepoc=100, addbias=True, l1=0, l2=0):
        '''
        Treina um modelo logístico com Gradiente Descendente.
        Entradas:
        x: Matriz numpy com entradas (com ou sem bias)
        y: Vetor numpy com saídas
        lrate: Learning rate (default = 0.001)
        nepoc: Número de iterações ou epochs (default = 100)
        addbias: True: acrescenta coluna de 1s no final de X (default = True)
        l1: Lambda 1 (L1 regularization) (default = 0)
        l2: Lambda 2 (L2 regularization) (default = 0)
        Retorna:
        self: Instância de si mesmo
        
        '''
        self.X = x              # X
        self.Y = y              # Y
        self.learnrate = lrate  # Learning Rate
        self.epochs = nepoc     # Epochs
        self.bias = addbias     # add bias
        self.lambda1 = l1       # Lambda 1 (L1 regularization)
        self.lambda2 = l2       # Lambda 2 (L2 regularization)
        
        self.n = self.X.shape[0]    # Número de Amostras
        if self.bias:
            self.X = np.c_[self.X, np.ones(self.n)]
        self.d = self.X.shape[1]    # Número de Coeficientes
        
        # J = - SUM(Y_i * log(p_hat_i) + (1 - Y_i)*(1 - p_hat_i))
        # p_hat = sigmoid(z)
        # z = X*w
        # Jreg = J + 2*l1*|w| + l2*wt*w
        # dJ/dw = Xt*(Y - p_hat) + 2*l1*sign(w) + 2*l2*wt
        # Gradient Descent: w = w - (1/2)*learnrate*dJ/dw
        self.w = np.ones(self.d)/self.d
        for i in range(self.epochs):
            self.p_hat = self.sigmoid(self.z(self.X, self.w))
            error = self.Y - self.p_hat
            self.w += (self.X.T.dot(error) - self.lambda1*np.sign(self.w.T) - 
                       self.lambda2*self.w.T) * self.learnrate
        
        return self
    
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-1.0*z))
    
    def z(self, x, w):
        return x.dot(w)
    
    def predict(self, x):
        '''
        Realiza previsões com modelo Logístico.
        Entrada: 
        x: Matriz numpy com entradas
        Saída:
        yhat: yhat = round( x * self.w )
        '''
        if ((x.size / x.shape[0]) == (self.d - 1)):
            X = np.c_[x, np.ones(x.shape[0])]
        else:
            X = x
        return np.round(self.sigmoid(self.z(X, self.w)))
    
    def plot(self):
        '''
        Gera gráfico das entradas e do modelo Logístico.
        Apenas para Regressão Logística com duas entradas (X1, X2).
        '''
        plt.scatter(self.X[:,0], self.X[:,1], c=self.Y, label='Y')
        x1min = self.X[:,0].min()
        x1max = self.X[:,0].max()
        x1 = np.linspace(0.9*x1min, 0.9*x1max, 10)
        x2 = -(self.w[2] + self.w[0]*x1) / self.w[1]
        plt.plot(x1, x2, c='red', label='y_hat')
        plt.legend()
        plt.show()
        
    def classification_error(self):
        return sum(self.Y != self.predict(self.X))/self.n * 100