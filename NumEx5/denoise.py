__author__ = 'julenka'

### Apply denoising to jingle
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
mat_contents = sio.loadmat('jingle.mat')

jingle = mat_contents['jingle']
jingle_len = len(jingle[0])

wavfile.write('jingle.wav', 44100, jingle[0])

sigma = 0.01
noise = sigma * np.random.randn(255000)

jingle_noisy = jingle + noise
jingle_noisy = jingle_noisy.reshape(255000,)
wavfile.write('jingle-noisy.wav', 44100, jingle_noisy)

M = 10
lbd = float(M-1)/float(M)
h = (1-lbd)*pow(lbd,np.arange(100)) # constructs the impulse response of the system of length 100

jingle_conv = np.convolve(jingle_noisy, h, 'valid')
wavfile.write('jingle-denoised.wav', 44100, jingle_conv)

M = 10
lbd = float(M-1)/M
h = (1-lbd) * pow(lbd, arange(100))

L = len(h)
N_J = jingle_len

from scipy import fftpack as f
JingleNoisy = f.fft(jingle_noisy, N_J-L+1)
H = f.fft(h,N_J-L+1)

normFreqJ = np.arange(N_J-L+1, dtype=float)/N_J-L+1
