#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pylab as plt
from scipy.stats import norm as awgn
from numpy.linalg import inv
import math


# In[220]:


def get_submatrix(offset, Z):
    if offset == -1: return np.zeros((Z, Z), dtype = bool)
    offset %= Z
    if offset == 0: return np.eye(Z, dtype = bool)
    return sp.diags([np.ones(offset), np.ones(Z - offset)],
                    [offset - Z, offset], dtype = bool).toarray()

# In[577]:


#5G method https://ieeexplore.ieee.org/document/8882079

def H_disassemble(H, Z):
    m, n = H.shape
    g = 4*Z

    C = H[:g, :(n-m)]
    D = H[:g, (n-m):(n-m)+(g)]
    E = H[:g, (n-m)+(g):]
    A = H[g:, :(n-m)]
    B = H[g:, (n-m):(n-m)+(g)]
    T = H[g:, (n-m)+(g):]

    assert C.shape[1] + D.shape[1] + E.shape[1] == n
    assert A.shape[1] + B.shape[1] + T.shape[1] == n
    assert C.shape[0] + A.shape[0] == m
    assert D.shape[0] + B.shape[0] == m
    assert E.shape[0] + T.shape[0] == m
    assert sum(sum(E)) == 0
    assert sum(sum(np.eye(m - g) - T)) == 0

    D_inv = inv(D.astype(int))
    F = (D_inv @ C % 2).astype(bool)
    
    return F, A, B

def get_codevector(a, F, A, B, seed = 2021):
    p_1 = F @ a.T % 2
    p_2 = (A @ a.T + B @ p_1) % 2

    d = np.hstack((a, p_1, p_2))
    return d #len(d) = 68 * Z


# In[578]:


# rate matching and redundancy version support

def get_shift(rv):
    # N_cb = 66 * Z and no N_ref
    if rv == 0: shift = 0
    elif rv == 1: shift = 17 * Z
    elif rv == 2: shift = 33 * Z
    elif rv == 3: shift = 56 * Z
    else: print('unkknown rv'); shift = 0
    return shift

def convolution(d, Z, shift, L):
    d_cut = d[2*Z:] #len(d) = 66 * Z
    if (L + shift < len(d_cut)):
        v = d[shift: shift + L]
    else:
        n_repeat = max(math.floor((L - (len(d_cut) - shift)) / len(d_cut)), 0)
        last_len =  (L - (len(d_cut) - shift)) % len(d_cut) if (L - (len(d_cut) - shift) > 0)  else 0
        #print(shift, n_repeat, '*', len(d_cut), last_len)
        v = np.hstack((d_cut[shift:], list(d_cut) * n_repeat, d_cut[:last_len]))
    return v
    
def involution(l, Z, shift):
    L = len(l)
    r = np.zeros(66 * Z)
    

    exp_start = shift
    exp_end = len(r) - ((L - (len(r) - shift)) % len(r))
    l_exp = np.hstack((np.zeros(exp_start), l, np.zeros(exp_end)))
    n_repeat = int(len(l_exp) / len(r)) # expanded with start and end

    for k in range(n_repeat):
        r = np.add(r, np.array(l_exp[k*len(r): (k+1)*len(r)]))
    
    r = np.hstack((np.zeros(2*Z), r))
    return r


# In[584]:


def demodulate(b):
    u = 0.5 * (1 - np.sign(b))
    return u.astype(bool)

def modulate(v):
    return 1 - 2*v

def encoder(H, Z, SNR, d, L = 300, rv = 1, seed = 2021):
    # SNR: Signal-Noise Ratio. SNR = 10 * lg(1 / variance) =  -20 lg (sigma) in dB.
    # BPSK modulation: "0" -> +1., "1" -> -1.
    
    shift = get_shift(rv = rv)
    v = convolution(d, Z, shift, L = L)
    
    x = modulate(v)
    sigma = 10 ** (- SNR / 20)
    np.random.seed(seed = seed)
    e = awgn.rvs(size = x.shape, scale = sigma)
    
    y = x + e
    return y

def decoder(H, y, Z, SNR, rv = 1, alg = 'MSA'):
    sigma = 10 ** (- SNR / 20)
    l = 2 * y / (sigma**2)
    
    shift = get_shift(rv = rv)
    r = involution(l, Z, shift)
    r = np.where(r != 0, r, awgn.rvs(scale = sigma)) #workaround

    if alg == 'MSA':
        b = MSA(H, r, SNR = SNR)
    elif alg == 'SPA':
        b = SPA(H, r, SNR = SNR)
    else:
        print('unknown alg'); return np.zeros(y.shape).astype(bool)
    
    u = demodulate(b)
    
    #s = H @ u % 2
    #if sum(s) != 0: print('probably there is an error')
    return u


# In[615]:


@np.vectorize
def custom_llr(val):
    if abs(val) == 1.: return val
    return np.log((1 + val)/(1 - val))

@np.vectorize
def custom_tanh(M):
    return np.tanh(M)

def SPA(H, r, SNR, max_iter = 10, threshold = 1.):
    
    M = np.copy(H).astype(float)
    E = np.zeros(H.shape)
    H_inv = (H + np.ones(np.shape(H))) %2

    for i in range(H.shape[0]):
        M[i, :] = r * H[i, :]

    for k in range(max_iter):
        #print(k)
        
        M = custom_tanh(M/2) + H_inv
        t = np.prod(M, axis = 1) 
        for j in range(H.shape[1]):
            E[:, j] = t * H[:, j]
        E[:,:] = E[:, :] / M[:, :]
        E = H*E
        E = custom_llr(E)

        belief = np.sum(E, axis = 0) + r
        #print(belief)
        M = (H != 0) * (belief - E)
        #print(k, sum(H @ demodulate(belief) % 2))
        if (sum(H @ demodulate(belief) % 2) == 0):
            return belief
        if (np.prod(np.abs(belief) > threshold)):
            return belief
        
    return belief

def MSA(H, r, SNR, max_iter = 30, threshold = 1.):
    belief = r
    
    L = np.copy(H).astype(float)
    #initialization
    for i in range(H.shape[0]):
        L[i, :] = r * H[i, :]

    for k in range(max_iter):
        #row step
        for i in range(L.shape[0]):
            magnitudes = sorted(set(abs(L[i])), reverse = False)
            magnitudes.remove(0)
            if len(list(magnitudes)) == 1:
                min1, min2 = magnitudes[0], magnitudes[0]
            else:
                min1, min2 = list(magnitudes)[:2]
            argmin1 = np.where(abs(L[i]) == min1)[0]
            signs = np.sign(L[i])
            parity = np.sum(signs < 0) % 2

            L[i] = (H[i] != 0) * min1
            L[i][argmin1] = min2
            L[i] *= signs * (-1) ** parity

        #column step
        belief = np.sum(L, axis = 0) + r
        L = (H != 0) * (belief - L)
        
        #print(sum(H @ demodulate(belief) % 2), k)
        if (sum(H @ demodulate(belief) % 2) == 0):
            return belief
        if (np.prod(np.abs(belief) > threshold)):
            return belief
        
    return belief


# In[620]:


def parse_H(n_set, Z):
    filler = np.vectorize(get_submatrix, otypes = [np.ndarray], signature = '()->(n,n)', excluded = (1,))
    matrix = np.loadtxt('matrices/R1-1711982_BG1_set{}.csv'.format(n_set), dtype = int)
    H = np.block([[x for x in row] for row in filler(matrix, Z)])
    F, A, B = H_disassemble(H, Z)
    return H, F, A, B

class experiment:
    def __init__(self, n_set = 1, Z = 6, R = 1/3, alg = 'MSA'):
        self.n_set = n_set
        self.Z = Z
        self.L = int(22*self.Z/R)
        self.alg = alg
        self.H, self.F, self.A, self.B = parse_H(self.n_set, self.Z)
        
    def run(self, seed, rv):
        np.random.seed(seed = seed)
        tx = np.random.randint(low = 0, high = 2, size = self.F.shape[1])
        d = get_codevector(tx, self.F, self.A, self.B)

        y = encoder(self.H, self.Z, self.SNR, d, L = self.L, rv = rv, seed = seed)
        u = decoder(self.H, y, self.Z, self.SNR, rv = rv, alg = self.alg)
        rx = u[:self.F.shape[1]]

        res = '{}\t{}\t{:.3f}\t{:.6f}\t{:.6f}\t{}\t{}\n'.format(self.SNR, rv, (len(tx)/self.L), abs(sum(d - u)/len(d)), abs(sum(rx - tx)/len(tx)), sum(d - u) == 0, sum(rx - tx) == 0)
        return res
        
    def set_SNR(self, SNR):
        self.SNR = SNR


# In[623]:

R = 1/4
rv = 0
exp = experiment(n_set = 1, Z = 12, R = R, alg = 'SPA')
f = open("results_{:.3f}_{}.txt".format(R, rv), "a")

for SNR in np.arange(0., 10.01, 0.25):
  exp.set_SNR(SNR)
  for seed in range(0, 1000, 1):
    res = exp.run(seed = seed, rv = rv)
    f.write(res)
    if (seed % 100 == 0) : print(res)
f.close()

