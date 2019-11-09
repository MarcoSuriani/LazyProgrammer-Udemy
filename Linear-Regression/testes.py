# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:21:29 2019

@author: Suriani
"""

from LinearRegressionFunc import LinearRegressionSuriani
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 20000
    
    ##  TESTE 1
    X = 10*np.random.rand(N)
    k = np.random.randn(N) - 10
    Y = 2*X - X**2 + k
    
    #modelo = LinearRegressionSuriani()
    #modelo.fit(X, Y, 0.000001, 5000000)
    #print('TESTE #01')
    #print('Y = 5X - X²')
    #print(modelo.w)
    #modelo.plot()
    #print('\n')
    
    ##  TESTE 2
    XC = np.c_[X, X**2]
    
    modelo = LinearRegressionSuriani()
    modelo.fit(XC, Y, 0.001, 300000)
    print('TESTE #02')
    print('Y = 5X - X²')
    print('Acrescenta coluna x² ao modelo')
    print(modelo.w)
    print(modelo.X.shape)
    modelo.plot()
    plt.plot(modelo.J[-50:])
    plt.show()
    print('\n')
    
    xm = modelo.X
    wa = np.dot(np.linalg.inv(xm.T.dot(xm)), xm.T.dot(modelo.Y))
    print(wa)