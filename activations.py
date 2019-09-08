# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:05:03 2019

@author: Rafael Rocha
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_func(x, a=1):
    y = 1/(1+np.exp(-x*a))
    dy = y*(1-y)
    return y, dy

def tanh_func(x):    
    y = np.sinh(x)/np.cosh(x)
    dy = 1 - np.power(y,2)
    return y, dy

def relu_func(x):
    y = np.maximum(x,0)
    dy = np.heaviside(x,x)
    return y, dy

x = np.linspace(-10,10)
y, dy = sigmoid_func(x, a=1)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
#plt.plot(x, sigmoid_func(x, a=2), '--')
#plt.plot(x, sigmoid_func(x, a=32), '.')
plt.ylabel(r'$\varphi$(v)')
plt.xlabel('v')
plt.title('Sigmoide')
#plt.axvline(0, color='black')
plt.grid()

y, dy = tanh_func(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
plt.xlabel('v')
plt.ylabel(r'$\varphi$(v)')
plt.title('Tanh')
plt.grid()

y, dy = relu_func(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ReLU')
plt.grid()