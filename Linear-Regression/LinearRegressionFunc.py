# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:06:14 2018

@author: Suriani
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionSuriani:
    def fit(self, x, y, lrate=0.001, nepoc=100, l1=0, l2=0):
        '''
        Treina um modelo linear com Gradiente Descendente.
        Sempre acrescenta uma coluna de bias (1s) no final.
        
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
        self.n = int(x.shape[0])    # Número de Amostras
        self.X = np.c_[x, np.ones(self.n)]      # Matriz X com bias   
        ind = self.X[:,0].argsort()
        self.X = self.X[ind]
        #self.Y = np.zeros((self.n, 1))
        #self.Y[:,0] = y         # Y
        self.Y = y[ind]
        self.learnrate = lrate  # Learning Rate
        self.epochs = nepoc     # Epochs
        self.lambda1 = l1       # Lambda 1 (L1 regularization)
        self.lambda2 = l2       # Lambda 2 (L2 regularization)
        
        
        # Lote é 1/10 do número de amostras, no máximo 50 pontos
        self.b = int(min(np.ceil(self.n / 10), 50))
        self.d = self.X.shape[1]    # Número de Coeficientes
        
        # J = (Y - Xw)t * (Y - Xw) + 2*l1*|w| + l2*wt*w
        # dJ/dw = 2*Xt*(Yhat - Y) + 2*l1*sign(w) + 2*l2*w
        # Gradient Descent: w = w - (1/2)*learnrate*dJ/dw
        self.w = np.ones(self.d) / self.d
        self.J = []

        for i in range(self.epochs):
            # Cria lotes (batches) aleatórios de b pontos
            lote = tuple(np.random.randint(0, self.n, self.b))
            XB = self.X[lote ,:]
            YB = self.Y.take(lote)
            yhat = XB.dot(self.w)
            error = (yhat - YB)/self.n
            dJdw = XB.T.dot(error)
            if max(abs(dJdw)) < 0.0001:
                print('break at: ', i)
                break
            self.w -= 2 * self.learnrate * (XB.T.dot(error) + 
                                            self.lambda1*np.sign(self.w) + 
                                            self.lambda2*self.w)
            E = self.X.dot(self.w)/self.n
            self.J.append( E.T.dot(E) )
            #print(self.w.shape)
            #print(self.w)
        print(XB.T.dot(error))
        return self
    
    def predict(self, x, addbias=True):
        '''
        Realiza previsões com modelo Linear.
        Entrada: 
        x: Matriz numpy com entradas
        Saída:
        yhat: yhat = x * self.w
        '''
        if addbias:
            X = np.c_[x, np.ones(x.shape[0])]
        else:
            X = x

        return X.dot(self.w)
    
    def plot(self):
        '''
        Gera gráfico das entradas e do modelo linear.
        Apenas para Regressão Linear com uma variável (y = f(x)).
        '''
        
        plt.scatter(self.X[:,0], self.Y, c='blue', label='Y')
        plt.plot(self.X[:,0],  self.predict(self.X, False), c='red', 
                 label='y_hat')
        plt.legend()
        plt.show()