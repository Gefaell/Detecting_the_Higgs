# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:24:00 2021

@author: gefae
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.special import erf

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
    
    return S

def Simpson_int_err(y, a, b, epsi, err):
    '''
    Numerical integration method using Simpson's rule
    y: function to be integrated
    a: lower limit of integration
    b: upper limit of integration
    '''
    h = b - a
    I1 = 0.5 * h * (y(a, err) + y(b, err))
    h = (b - a)/2
    I2 = h * (0.5*y(a, err) + y((a+b)/2, err) + 0.5*y(b, err))
    S = (4/3)*I2 - (1/3)*I1
    
    i = 2
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
                    sumterms = y(a + h, err)
                else:
                    sumterms += y(a + num[l]*h, err)
        
        I2 = 0.5*I2 + h*sumterms
        S = (4/3)*I2 - (1/3)*I1
        param = abs((S-S1)/S1)
    
    return S

int1_Simpson = Simpson_int(Gauss, 0, 1, 1e-6)

true_val = erf(1/np.sqrt(2))/2
acc = abs(int1_Simpson-true_val)
    
print("Accuracy of Simpson's rule:", acc)

#%% Checking with improper integrals

def Gauss2(t):
    y = (1/(t**2))*(1/np.sqrt(2*np.pi))*np.exp((-(1/t)**2)/2)
    return y

int3 = Simpson_int(Gauss2, 1e-4, 1, 1e-6) #equivalent to integral from 1 to infinity of Gauss1
int4 = 1e-4*Gauss2((1e-4)/2)
int11_Simpson = int3 + int4

int5 = Simpson_int(Gauss2, 1e-4, 1000, 1e-6) #equivalent to integral from 0 to infinity of Gauss1
int6 = 1e-4*Gauss2((1e-4)/2)
int12_Simpson = int5 + int6

total = int12_Simpson - int11_Simpson
true_val2 = erf(1/np.sqrt(2))/2
acc2 = abs(total-true_val2)

print("Accuracy of Simpson's rule using improper integrals:", acc2)

#%% Detecting the Higgs

def backg(m):
    A = 1500 #(GeV/c^2)^-1
    k = 20 #GeV/c^2
    mH = 125.1 #GeV/c^2
    y = A*np.exp(-(m-mH)/k)
    return y

def signal(m):
    mH = 125.1 #GeV/c^2
    RMS = 1.4 #GeV/c^2
    N = 470
    s = N*(1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(m-mH)**2)/(2*RMS**2))
    return s

def significance(ml,mu):
    
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal, ml, mu,1e-6)
    S = NH/np.sqrt(NB)
    
    return S

m = np.arange(110, 140, 1e-2)
plt.figure(1)
plt.plot(m, backg(m) + signal(m), 'darkblue', label = 'Detections')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Number per GeV/c^2')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

S0 = significance(118.0,122.0)
S1 = significance(120.3,129.7)
S2 = significance(122.3,127.7)
S3 = significance(123.3,126.7)
S4 = significance(124.3,125.7)
S5 = significance(128.0,132.0)

#%%
    
def grad_method(y):
    
    #Initial guesses obtained after running it once with random numbers and lower accuracy
    ml = 123.23
    mu = 127.15
    
    h = 1e-3
    alpha = 1e-2
    param = 1
    while param != 0:
        dy_ml = (y(ml + h, mu) - y(ml - h, mu))/(2*h)
        dy_mu = (y(ml, mu + h) - y(ml, mu - h))/(2*h)
        
        ml += alpha*dy_ml
        mu += alpha*dy_mu
        f1 = y(ml,mu)
        
        dy_ml = (y(ml + h, mu) - y(ml - h, mu))/(2*h)
        dy_mu = (y(ml, mu + h) - y(ml, mu - h))/(2*h)
        
        ml += alpha*dy_ml
        mu += alpha*dy_mu
        f2 = y(ml,mu)
        
        if f1 > f2:
            param = 0
    
    h = 1e-6
    alpha = 1e-2
    param = 1
    while param != 0:
        dy_ml = (y(ml + h, mu) - y(ml - h, mu))/(2*h)
        dy_mu = (y(ml, mu + h) - y(ml, mu - h))/(2*h)
        
        ml += alpha*dy_ml
        mu += alpha*dy_mu
        f1 = y(ml,mu)
        
        dy_ml = (y(ml + h, mu) - y(ml - h, mu))/(2*h)
        dy_mu = (y(ml, mu + h) - y(ml, mu - h))/(2*h)
        
        ml += alpha*dy_ml
        mu += alpha*dy_mu
        f2 = y(ml,mu)
        
        if f1 > f2:
            param = 0
        
    return np.array([ml,mu])

maxi = grad_method(significance)

max_sig = significance(maxi[0], maxi[1])
NB = Simpson_int(backg, maxi[0], maxi[1],1e-6)
NH = Simpson_int(signal, maxi[0], maxi[1],1e-6)
print("Maximum significance:", max_sig)
print("Optimal selection cuts:", maxi[0], ', ', maxi[1])
print('Average number of Higgs detections:', NH)
print('Average number of background detections:', NB)

max_sig2 = significance(maxi[0]-1e-6, maxi[1]+1e-6)
max_sig3 = significance(maxi[0]+1e-6, maxi[1]-1e-6)
max_sig4 = significance(maxi[0]-1e-6, maxi[1]-1e-6)
max_sig5 = significance(maxi[0]+1e-6, maxi[1]+1e-6)

#%%

#plot significance in terms of ml and mu to show that it is smooth
#and that the gradient method doesn't get stuck

ml_arr = np.linspace(110,150,100)
mu_arr = np.linspace(110,150,100)
y, x = np.meshgrid(ml_arr, mu_arr)
sig_i = []
sig = []

ml_i = []
ml_tot = []

mu_i = []
mu_tot = []

for i in range(len(ml_arr)):
    cont = 0
    for j in range(len(mu_arr)):
        if ml_arr[i] < mu_arr[j]:
            cont += 1
            s = significance(ml_arr[i], mu_arr[j])
            sig_i.append(s)
            if j == len(mu_arr)-1:
                c = 1
            if j != len(ml_arr)-1:
                c = 0
            if c == 1:
                ml_i.append([ml_arr[i]]*(len(mu_arr)-1))
                mu_i.append(mu_arr[1:])
                ml_j = np.array(copy.deepcopy(ml_i[0]))
                mu_j = np.array(copy.deepcopy(mu_i[0]))
                sig_j = np.concatenate((np.zeros(len(ml_arr)-cont),np.array(copy.deepcopy(sig_i))),axis=None)
                ml_tot.append(ml_j)
                mu_tot.append(mu_j)
                sig.append(sig_j)
                ml_i.clear()
                mu_i.clear()
                sig_i.clear()

z = np.concatenate((np.array(sig),np.zeros((1,len(ml_arr)))),axis=0)

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_ylabel("mu", labelpad=15.0)
ax.set_xlabel("ml", labelpad=15.0)
ax.set_zlabel("Significance", labelpad=15.0, rotation=180)
threeD_plot = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0.0, 5.2)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.zaxis.set_major_locator(LinearLocator(10))

fig.colorbar(threeD_plot, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()


#%%

#470 yields a maximum significance of ~5, so any value lower than 470 won't satisfy 5 significance
#and any higher value will, so probability of detecting a 5-sigma signal is ~50%

#%% Error source 1

def signal_err1(m, err):
    mH = 125.1 #GeV/c^2
    RMS = 1.4 #GeV/c^2
    N = 470
    s = N*(1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(m-mH - err)**2)/(2*RMS**2))
    return s

def significance1(ml,mu):
    
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal_err1, ml, mu,1e-6)
    S = NH/np.sqrt(NB)
    
    return S

ml = 123.23751357
mu = 127.15848853

err_arr = np.arange(-0.2, 0.21, 1e-2)
NB = Simpson_int(backg, ml, mu,1e-6)
NH_err1 = []
for i in range(len(err_arr)):
    NH_err1.append(Simpson_int_err(signal_err1, ml, mu,1e-6,err_arr[i]))

NH_av = Simpson_int(signal, ml, mu,1e-6)

plt.figure(3)
plt.plot(err_arr+125.1, NH_err1, 'darkblue')
plt.axhline(NH_av, linestyle = '--')
plt.xlabel('Theoretical mass (GeV/c^2)')
plt.ylabel('Number of HIggs detections')
plt.tight_layout()
plt.grid()
plt.show()

#%% Error source 2

def signal_err2(m):
    mH = 124.5 #GeV/c^2
    RMS = 2.6 #GeV/c^2
    N = 470
    s = N*(1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(m-mH)**2)/(2*RMS**2))
    return s

def significance2(ml,mu):
    
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal_err2, ml, mu,1e-6)
    S = NH/np.sqrt(NB)
    
    return S

ml = 123.23751357
mu = 127.15848853
ml2 = 121.19480958251769 #grad_method(significance2)[0]
mu2 = 128.48118749085498 #grad_method(significance2)[1]

p = np.arange(0.0, 0.0401, 1e-4)
NB = Simpson_int(backg, ml, mu,1e-6)
int_a = (1-p)*Simpson_int(signal, ml, mu,1e-6) #unaffected photons
int_b = p*Simpson_int(signal_err2, ml2, mu2,1e-6) #interacting photons
NH_err2 = int_a + int_b

NH_av = Simpson_int(signal, ml, mu,1e-6)

plt.figure(4)
plt.plot(p*100, NH_err2, 'darkblue')
plt.axhline(NH_av, linestyle = '--')
plt.xlabel('Affected photons (%)')
plt.ylabel('Number of Higgs detections')
plt.tight_layout()
plt.grid()
plt.show()

#%% Error source 3

def signal_err3(m, err): 
    mH = 125.1 #GeV/c^2
    RMS = 1.4 #GeV/c^2
    N = 470
    s = N*(1+err)*(1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(m-mH)**2)/(2*RMS**2))
    return s

def significance3(ml,mu):
    
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal_err3, ml, mu,1e-6)
    S = NH/np.sqrt(NB)
    
    return S

ml = 123.23751357
mu = 127.15848853

err_arr = np.arange(-0.03, 0.031, 1e-2)
NB = Simpson_int(backg, ml, mu,1e-6)
NH_err3 = []
for i in range(len(err_arr)):
    NH_err3.append(Simpson_int_err(signal_err3, ml, mu,1e-6,err_arr[i]))

NH_av = Simpson_int(signal, ml, mu,1e-6)

plt.figure(5)
plt.plot(100*err_arr, NH_err3, 'darkblue')
plt.axhline(NH_av, linestyle = '--')
plt.xlabel('Error in the number of created Higgs (%)')
plt.ylabel('Number of Higgs detections')
plt.tight_layout()
plt.grid()
plt.show()

#%% Calculate the probability of detecting a five sigma signal:

# Calculate P(N >= NB + 5*sqrt(NB)) where N = NH + NB
# mu = NH + NB and std = sqrt(NH + NB) for the sum of two Poisson distributions
# N - NB will follow a Poisson distribution with mean NH and standard deviation sqrt(NH)
# In addition we have sources of error

def Gaussian_sum(N):
    ml = 123.23751357
    mu = 127.15848853
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal, ml, mu,1e-6)
    av = NH + NB #GeV/c^2
    RMS = np.sqrt(NH + NB) #GeV/c^2
    s = (1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(N-av)**2)/(2*RMS**2))
    return s

NH_av = Simpson_int(signal, ml, mu,1e-6)
NB = Simpson_int(backg, ml, mu,1e-6)
val1 = NB + 5*np.sqrt(NB)
RMS = np.sqrt(NH_av + NB)

prob_5sig = Simpson_int(Gaussian_sum, val1, NB+NH_av+6*RMS, 1e-6)
print('Probability of detecting a 5-sigma signal without taking into account the errors:', 100*prob_5sig, '%')

tot_err = np.sqrt(NH_av + (NH_av-NH_err1[0])**2 + (NH_av-NH_err2[-1])**2 + (NH_av-NH_err3[0])**2)
print('Total error in the number of Higgs detections (NH): +/-', tot_err)

def Gaussian_sum_errneg(N):
    ml = 123.23751357
    mu = 127.15848853
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal, ml, mu,1e-6) - tot_err
    av = NH + NB #GeV/c^2
    RMS = np.sqrt(NH + NB) #GeV/c^2
    s = ((1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(N-av)**2)/(2*RMS**2)))
    return s

def Gaussian_sum_errpos(N):
    ml = 123.23751357
    mu = 127.15848853
    NB = Simpson_int(backg, ml, mu,1e-6)
    NH = Simpson_int(signal, ml, mu,1e-6) + tot_err
    av = NH + NB #GeV/c^2
    RMS = np.sqrt(NH + NB) #GeV/c^2
    s = ((1/(RMS*np.sqrt(2*np.pi)))*np.exp((-(N-av)**2)/(2*RMS**2)))
    return s

prob_5signeg = Simpson_int(Gaussian_sum_errneg, val1, NB+NH_av+6*RMS, 1e-6)
prob_5sigpos = Simpson_int(Gaussian_sum_errpos, val1, NB+NH_av+6*RMS, 1e-6)

print('Lower bound of probability of detecting a 5-sigma signal:', 100*prob_5signeg, '%')
print('Upper bound of probability of detecting a 5-sigma signal:', 100*prob_5sigpos, '%')













