__author__ = 'julenka'

#  Construct the impulse response of the system of length 100
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt

M = 10
lbd = float((M-1))/float(M)
h = (1-lbd)*pow(lbd,np.arange(100))

#  Generate two tones
y = np.linspace(0,999,1000)
x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*80)/1000)
x = np.append(x1,x2)
# x = x1

# Generate the noise
sigma3 = 0.1
noise = sigma3 * np.random.randn(x.shape[0])

# Add noise to signal
xNoisy = noise + x

N = len(xNoisy)
L = len(h)

# DFT/DFS of noisy signal and impulse response
from scipy import fftpack as f
XNoisy = f.fft(xNoisy,N-L+1)
H = f.fft(h,N-L+1)

normFrequ = np.arange(N-L+1,dtype=float)/(float(N-L+1)) # To plot vs the normalized frequencies

plt.subplot(2,1,1)
plt.plot(normFrequ,abs(XNoisy))
plt.xlabel('Normalized Frequencies')
plt.ylabel('|XNoisy|')
plt.subplot(2,1,2)
plt.plot(normFrequ,abs(H))
plt.xlabel('Normalized Frequencies')
plt.ylabel('|H|')
plt.show()

ABS = H * XNoisy

plt.subplot(2,1,1)
plt.plot(normFrequ,abs(ABS))
plt.xlabel('Normalized Frequencies')
plt.ylabel('|H * XNoisy|')

plt.subplot(2,1,2)
plt.plot(f.ifft(ABS))
plt.xlabel('sampled')
plt.ylabel('IFFT(}|H * XNoisy|')
plt.show()

