# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:07:37 2018

@author: Suriani
"""

from LinearRegressionFunc import LinearRegressionSuriani
import numpy as np

if __name__ == '__main__':
    N = 600
    
    ##  TESTE 1
    X = 10*np.random.rand(N)
    Y = 2.5*X + .75 + np.random.randn(N)

    print('TESTE #01')
    print('Y = 2.5X + .75')
    modelo = LinearRegressionSuriani()
    modelo.fit(X, Y, 0.001, 10000)
    print(modelo.w)
    modelo.plot()
    print('\n')
    
    ##  TESTE 2
    Y[N-2] += 20
    Y[N-1] += 20

    print('TESTE #02')
    print('Y = 2.5X + .75 com outliers')    
    print('sem regularização L2')
    modelo = LinearRegressionSuriani()
    modelo.fit(X, Y, 0.001, 100000, 0, 0)
    print(modelo.w) 
    print('\n')
    
    ##  TESTE 3
    print('TESTE #03')
    print('Y = 2.5X + .75 com outliers')
    print('com regularização L2')
    modelo = LinearRegressionSuriani()
    modelo.fit(X, Y, 0.001, 200000, 0, 1)
    print(modelo.w) 
    print('\n')
    
    ##  TESTE 4
    X2 = 10*np.random.rand(N)
    X3 = 10*np.random.rand(N)
    X4 = 10*np.random.rand(N)
    Y = 2.5*X - 1.5*X2 + 0.5*np.random.randn(N)
    
    XC = np.c_[X, X2, X3, X4]

    print('TESTE #04')
    print('Y = 2.5X1 - 1.5*X2 + 0*X3 + 0*X4')
    print('sem regularização')
    modelo = LinearRegressionSuriani()
    modelo.fit(XC, Y, 0.001, 200000)
    print(modelo.w) 
    print('\n')
    
    ##  TESTE 5
    print('TESTE #05')
    print('Y = 2.5X1 - 1.5*X2 + 0*X3 + 0*X4')
    print('com regularização L1')
    modelo = LinearRegressionSuriani()
    modelo.fit(XC, Y, 0.001, 200000, 0.1, 0)
    print(modelo.w) 
    print('\n')
    
    ##  TESTE 6
    print('TESTE #06')
    print('Y = 2.5X1 - 1.5*X2 + 0*X3 + 0*X4')
    print('com regularização L1 e L2')
    modelo = LinearRegressionSuriani()
    modelo.fit(XC, Y, 0.0001, 200000, 1, 1) 
    print(modelo.w) 
    print('\n')
    
    ##  TESTE 7
    Y = 2.5*X - X**2 + .75 + np.random.randn(N)
    
    print('TESTE #07')
    print('Y = 2.5X - X² + .75')
    modelo = LinearRegressionSuriani()
    modelo.fit(X, Y, 0.001, 200000)
    print(modelo.w)
    modelo.plot()
    print('\n')
    
    ##  TESTE 8
    XC = np.c_[X, X**2]
    
    print('TESTE #08')
    print('Y = 2.5X - X² + .75')
    print('Acrescenta coluna x² ao modelo')
    modelo = LinearRegressionSuriani()
    modelo.fit(XC, Y, 0.0001, 50000)
    print(modelo.w)
    modelo.plot()
    print('\n')