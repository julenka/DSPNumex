__author__ = 'julenka'
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt

## Noise
y = np.linspace(0,999,1000)
M = 10
lbd = float(M-1)/float(M)
h = (1-lbd)*pow(lbd,np.arange(100)) # constructs the impulse response of the system of length 100

sigma2 = 0.1    #Power of the noise
noise = sigma2*np.random.randn(2000)    # Gaussian noise

x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*80)/1000)
x = np.append(x1,x2)
xNoisy = noise + x # Noisy version of x
yConv = np.convolve(xNoisy,h,'valid')

plt.subplot(4,1,1)
plt.plot(np.arange(x.size),x)
plt.ylabel('x')
plt.xlim([825,1175])

plt.subplot(4,1,2)
plt.plot(np.arange(xNoisy.size),xNoisy)
plt.xlabel('Samples')
plt.ylabel('xNoisy')
plt.xlim([825,1175])

plt.subplot(4,1,3)
plt.ylabel('h')
plt.plot(h)

plt.subplot(4,1,4)
plt.ylabel('yConv')
plt.xlim([825,1175])
plt.plot(yConv)
plt.show()


