# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:19:25 2021

@author: gefae
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy.special import erf
import random

params = {
   'font.family' : 'serif',
   'axes.labelsize': 19,
   'font.size': 19,
   'legend.fontsize': 17,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'figure.figsize': [12, 12/1.618],
   'errorbar.capsize': 3,
   'lines.markersize': 8,
   'mathtext.default': 'regular'
   } 

plt.rcParams.update(params)

#%% Gaussian function

def Gauss(t):
    y = (1/np.sqrt(2*np.pi))*np.exp((-t**2)/2)
    return y

#%% SPECIFIC METHODS

print('SPECIFIC METHODS:')
print(
        )

#%% Trapezoidal rule

def trap_int(y, a, b):
    '''
    Numerical integration method using trapezoidal rule
    y: function to be integrated
    a: lower limit of integration
    b: upper limit of integration
    '''
    h = b - a
    I1 = 0.5 * h * (y(a) + y(b))
    h = (b - a)/2
    I2 = h * (0.5*y(a) + y((a+b)/2) + 0.5*y(b))
    
    i = 2
    #eps = np.finfo(float).eps
    param = abs((I2-I1)/I1)
    
    while param > 1e-10: 
        i += 1
        I1 = copy.deepcopy(I2)
        div = 2**(i-1)
        h = h/2
        
        num = np.arange(1, div, 1)
        for l in range(len(num)):
            if l%2 == 0:
                if l == 0:
                    sumterms = y(a + h)
                else:
                    sumterms += y(a + num[l]*h)
        I2 = 0.5*I2 + h*sumterms
        param = abs((I2-I1)/I1)
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I2-true_val)
    
    print('Trapezoidal stepsize:', h)
    print('Trapezoidal method accuracy:', acc)
    print('Trapezoidal number of steps:', i)
    print(
        )
    return I2

print('Integral from 0 to 1:')
int1_trapezoidal = trap_int(Gauss, 0, 1)
print('Integral from 0 to 5:')
int2_trapezoidal = trap_int(Gauss, 0, 5)

#%% Simpson's rule

def Simpson_int(y, a, b, epsi):
    '''
    Numerical integration method using Simpson's rule
    y: function to be integrated
    a: lower limit of integration
    b: upper limit of integration
    '''
    h = b - a
    I1 = 0.5 * h * (y(a) + y(b))
    h = (b - a)/2
    I2 = h * (0.5*y(a) + y((a+b)/2) + 0.5*y(b))
    S = (4/3)*I2 - (1/3)*I1
    
    i = 2
    #eps = np.finfo(float).eps
    param = 10**7
    
    while param > epsi:
        i += 1
        S1 = copy.deepcopy(S)
        I1 = copy.deepcopy(I2)
        div = 2**(i-1)
        h = h/2
        
        num = np.arange(1, div, 1)
        for l in range(len(num)):
            if l%2 == 0:
                if l == 0:
                    sumterms = y(a + h)
                else:
                    sumterms += y(a + num[l]*h)
        
        I2 = 0.5*I2 + h*sumterms
        S = (4/3)*I2 - (1/3)*I1
        param = abs((S-S1)/S1)
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(S-true_val)
    
    print('Simpson stepsize:', h)
    print('Simpson method accuracy:', acc)
    print('Simpson number of steps:', i)
    print(
        )
    return S

print('Integral from 0 to 1:')
int1_Simpson = Simpson_int(Gauss, 0, 1, 1e-15)
print('Integral from 0 to 5:')
int2_Simpson = Simpson_int(Gauss, 0, 5, 1e-15)

#%% ODE METHODS
print(
        )
print(
        )
print('ODE METHODS:')
print(
        )

#%% Euler method

def Euler_int(func, step, a, b):
    
    '''
    Numerical integration function using Euler's method
    func: function to be integrated
    step: step in numerical integration
    a: lower limit of integration
    b: upper limit of integration
    y: integral of func in a-b range
    '''
    
    t = [a]
    y = []
    n_steps = math.floor((b - a)/step)
    
    dy = [func(a)] #ODE to be solved is dy = func
    for i in range(0,n_steps+1):
        if i == 0:
            new_t = t[-1] + step
            new_y = dy[-1]*step
            new_dy = func(new_t)
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
        else:
            new_t = t[-1] + step
            new_y = y[-1] + dy[-1]*step
            new_dy = func(new_t)
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
    
    t = np.array(t)
    y = np.array(y)
    I = y[-1] - y[0]
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I-true_val)
    print('Euler stepsize:', step)
    print('Euler method accuracy:', acc)
    print(
        )
    
    return I

print('Integral from 0 to 1:')
int1_Euler = Euler_int(Gauss, 1e-6, 0, 1)
print('Integral from 0 to 5:')
int2_Euler = Euler_int(Gauss, 1e-6, 0, 5)

#%% AB2 method

def AB2_int(func, step, a, b):
    
    '''
    Numerical integration function using AB2 method
    func: function to be integrated
    step: step in numerical integration
    a: lower limit of integration
    b: upper limit of integration
    y: integral of func in a-b range
    '''
    
    t = [a]
    y = []
    n_steps = math.floor((b - a)/step)
    
    dy = [func(a)] #ODE to be solved is dy = func
    t.append(t[0] + step)
    y.append(dy[0]*step) #first values obtained with Euler method
    dy.append(func(t[-1]))
    for i in range(0,n_steps+1): 
        new_t = t[-1] + step
        new_y = y[-1] + 0.5*(3*dy[-1]-dy[-2])*step
        new_dy = func(new_t)
        t.append(new_t)
        y.append(new_y)
        dy.append(new_dy)
    
    t = np.array(t)
    y = np.array(y)
    I = y[-1] - y[0]
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I-true_val)
    print('AB2 stepsize:', step)
    print('AB2 method accuracy:', acc)
    print(
        )
    
    return I

print('Integral from 0 to 1:')
int1_AB2 = AB2_int(Gauss, 1e-6, 0, 1)
print('Integral from 0 to 5:')
int2_AB2 = AB2_int(Gauss, 1e-6, 0, 5)

#%% RK2 method

def RK2_int(func, step, a, b):
    
    '''
    Numerical integration function using RK2 method
    func: function to be integrated
    step: step in numerical integration
    a: lower limit of integration
    b: upper limit of integration
    y: integral of func in a-b range
    '''
    
    t = [a]
    y = []
    dy = [func(a)] #ODE to be solved is dy = func
    n_steps = math.floor((b - a)/step)
    alpha = 0.5
    
    for i in range(0,n_steps+2):
        if i == 0:
            new_t = t[-1] + step
            t_alpha = t[-1] + alpha*step
            dy_alpha = func(t_alpha)
            new_y = ((2*alpha - 1)/(2*alpha))*dy[-1]*step + dy_alpha*step
            new_dy = func(new_t)
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
        else:
            new_t = t[-1] + step
            t_alpha = t[-1] + alpha*step
            dy_alpha = func(t_alpha)
            new_y = y[-1] + ((2*alpha - 1)/(2*alpha))*dy[-1]*step + dy_alpha*step
            new_dy = func(new_t)
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
    
    t = np.array(t)
    y = np.array(y)
    I = y[-1] - y[0]
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I-true_val)
    print('RK2 stepsize:', step)
    print('RK2 method accuracy:', acc)
    print(
        )
    
    return I

print('Integral from 0 to 1:')
int1_RK2 = RK2_int(Gauss, 1e-6, 0, 1)
print('Integral from 0 to 5:')
int2_RK2 = RK2_int(Gauss, 1e-6, 0, 5)

#%% RK4 method

def RK4_int(func, step, a, b):
    
    '''
    Numerical integration function using RK4 method
    func: function to be integrated
    step: step in numerical integration
    a: lower limit of integration
    b: upper limit of integration
    y: integral of func in a-b range
    '''
    
    t = [a]
    y = []
    dy = [func(a)] #ODE to be solved is dy = func
    n_steps = math.floor((b - a)/step)
    
    for i in range(0,n_steps+2):
        if i == 0:
            new_t = t[-1] + step
            t_half = t[-1] + 0.5*step
            dy_b = func(t_half)
            dy_c = func(t_half) #dy_b and dy_c are the same because ODE methods
                                #for integration don't depend on u
            new_dy = func(new_t)
            new_y = (1/6)*(dy[-1] + 2*dy_b + 2*dy_c + new_dy)*step
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
        else:
            new_t = t[-1] + step
            t_half = t[-1] + 0.5*step
            dy_b = func(t_half)
            dy_c = func(t_half)
            new_dy = func(new_t)
            new_y = y[-1] + (1/6)*(dy[-1] + 2*dy_b + 2*dy_c + new_dy)*step
            t.append(new_t)
            y.append(new_y)
            dy.append(new_dy)
    
    t = np.array(t)
    y = np.array(y)
    I = y[-1] - y[0]
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I-true_val)
    print('RK4 stepsize:', step)
    print('RK4 method accuracy:', acc)
    print(
        )
    
    return I

print('Integral from 0 to 1:')
int1_RK4 = RK4_int(Gauss, 1e-6, 0, 1)
print('Integral from 0 to 5:')
int2_RK4 = RK4_int(Gauss, 1e-6, 0, 5)

#%% MONTE CARLO INTEGRATION

print(
        )
print(
        )
print('MONTE CARLO METHODS:')
print(
        )

def MC_int(y, a, b, N):
    '''
    Numerical integration method using Monte Carlo method
    y: function to be integrated
    a: lower limit of integration
    b: upper limit of integration
    N: number of random samples
    '''
    
    xi = random.uniform(a,b)
    sumvals = y(xi)
    L = b - a
    
    for i in range(2, N+2):
        x = random.uniform(a,b)
        f = y(x)
        sumvals += f
        I = L*sumvals/i
    
    true_val = erf(b/np.sqrt(2))/2
    acc = abs(I-true_val)
    
    print('Monte Carlo method accuracy:', acc)
    print('Monte Carlo number of random samples:', N)
    print(
        )
    return I

print('Integral from 0 to 1:')
int1_MC = MC_int(Gauss, 0, 1, 10**7)
print('Integral from 0 to 5:')
int2_MC = MC_int(Gauss, 0, 5, 10**7)












