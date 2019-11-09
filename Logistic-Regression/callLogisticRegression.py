# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:54:12 2019

@author: Suriani
"""

from LogisticRegressionFunc import LogisticRegressionSuriani
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    N4 = 250
    N2 = 2*N4
    N = 2*N2
    D = 2

    ##  TESTE 1
    d = 2
    X = np.concatenate((np.random.randn(N2, D)-d, np.random.randn(N2, D)+d), 0)
    Y = np.array([0]*N2 + [1]*N2)
    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 500)
    print('TESTE #01')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')

    ##  TESTE 2
    Y = np.array([1]*N2 + [0]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 500)
    print('TESTE #02')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')
    
    ##  TESTE 3
    d = 1
    X = np.concatenate((np.random.randn(N2, D)-d, np.random.randn(N2, D)+d), 0)
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 500)
    print('TESTE #03')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')
    
    ##  TESTE 4
    d = 1.5
    X1 = np.concatenate((np.random.randn(N4,1)-d, 
                         np.random.randn(N4,1)+d,
                         np.random.randn(N4,1)-d,
                         np.random.randn(N4,1)+d), 0)
    X2 = np.concatenate((np.random.randn(N4,1)+d, 
                         np.random.randn(N4,1)-d,
                         np.random.randn(N4,1)-d,
                         np.random.randn(N4,1)+d), 0)
    X = np.concatenate((X1, X2), 1)                        
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #04')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')
    
    ##  TESTE 5
    X = np.concatenate((X1, X2, X1*X2), 1)                        
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #05')
    print('Acrescenta termo X1*X2')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    
    sp = np.linspace(-10,10,100)
    XT = np.zeros((100**2,3))
    for i, x1 in enumerate(sp):
        for j, x2 in enumerate(sp):
            XT[100*i+j] = (x1, x2, x1*x2)
    print('\nVisualizando a classificação:')
    plt.scatter(XT[:,0], XT[:,1], c=modelo.predict(XT))
    plt.show()
    print('\n')
    
    ##  TESTE 6
    X = X1*X2                        
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #06')
    print('Apenas termo X1*X2')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    print('\n')
    
    ##  TESTE 7
    r = np.concatenate((np.random.randn(N2,1) + 5,
                        np.random.randn(N2,1) + 7.5), 0)
    
    theta = np.random.rand(N,1) * 2 * np.pi
    
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    X = np.concatenate((x1, x2), 1)                   
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #07')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')
    
    ##  TESTE 8
    X = np.concatenate((x1, x2, (x1*x1 + x2*x2)**0.5), 1)  
    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #08')
    print('Acrescenta termo (X1^2 + X2^2)^(1/2)')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    
    sp = np.linspace(-10,10,100)
    XT = np.zeros((100**2,3))
    for i, x1 in enumerate(sp):
        for j, x2 in enumerate(sp):
            XT[100*i+j] = (x1, x2, (x1**2 + x2**2)**0.5)
    print('\nVisualizando a classificação:')
    plt.scatter(XT[:,0], XT[:,1], c=modelo.predict(XT))
    plt.show()
    
    print('\n')
    
    ##  TESTE 9
    X = np.concatenate((x1, x2, x1*x2, (x1*x1 + x2*x2)**0.5), 1)  
    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 5000)
    print('TESTE #09')
    print('Acrescenta termos (x1*x2) e (X1^2 + X2^2)^(1/2)')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    print('\n')
    
    ##  TESTE 10
    d = 1.75
    X = np.concatenate((np.random.randn(N2, D)-d, np.random.randn(N2, D)+d), 0)
    X = np.concatenate((X, np.random.randn(N,1)), 1)
    Y = np.array([0]*N2 + [1]*N2)

    
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 500)
    print('TESTE #10')
    print('X3 = ruido, sem regularização')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    modelo.plot()
    print('\n')
    
    ##  TESTE 11
    modelo = LogisticRegressionSuriani()
    modelo.fit(X, Y, 0.001, 500, True, 10, 0)
    print('TESTE #11')
    print('X3 = ruido, com regularização L1')
    print(modelo.w)
    print('Classification Error = {:.2f}%'.\
          format(modelo.classification_error()))
    print('\n')