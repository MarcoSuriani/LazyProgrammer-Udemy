# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:54:59 2019

@author: Suriani
"""

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N, D))
X[:,0] = 1 # bias term
X[:,1] = range(0,N)
X[:,2] = X[:,1]**2
Y = 2*X[:,2] - X[:,1]

# print X so you know what it looks like
print(type(X))
print("X:", X)

# won't work!
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# let's try gradient descent
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.00001
for t in range(200):
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*(X.T.dot(delta) + 10*np.sign(w))

  # find and store the cost
  mse = delta.dot(delta) / N
  costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot prediction vs target
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()