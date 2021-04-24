#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pylab as plt
from scipy.stats import norm as awgn


def get_submatrix(offset, Z):
    if offset == -1: return np.zeros((Z, Z), dtype = bool)
    offset %= Z
    if offset == 0: return np.eye(Z, dtype = bool)
    return sp.diags([np.ones(offset), np.ones(Z - offset)],
                    [offset - Z, offset], dtype = bool).toarray()


def my_permutation(B, permut, n1, n2, n3):
    v1 = B.take(n1, axis = 1)
    v2 = B.take(n2, axis = 1)
    index1, index2 = permut[n1], permut[n2]
    permut[n1], permut[n2] = index2, index1
    for i in range(v1.shape[0]):
        B[i][n2] = v1[i]
        B[i][n1] = v2[i]
        
    ones_in_n2 = list(np.where(v1 == 1)[0])
    ones_in_n2.remove(n3)
    for i in ones_in_n2:
        B[:][i] = (B[:][i] + B[:][n3]) % 2

        
def custom_gauss_elimination(H):
    B = np.copy(H)
    completed_indexes = [] 
    drop_indexes = []
    K = B.shape[0]
    N = B.shape[1]
    permut = np.arange(N)
    for k_iter in range(K):
        row = B.take(k_iter, axis = 0)
        if(sum(row) != 0):
            i = 0
            while(row[i] != 1):
                i+=1
            my_permutation(B, permut, i, N - K + k_iter, k_iter)
            completed_indexes.append(N - K + k_iter)
        else:
            drop_indexes.append(k_iter)
            
    if not drop_indexes: 
        return B, permut
    else:
        i = 0
        for drop in drop_indexes:
            H = np.delete(H, drop - i, axis=0)
            i+=1
        print("try again...")
        custom_gauss_elimination(H)
        
        
def H2G(H):
    B, permut = custom_gauss_elimination(H)
    assert B[:, -B.shape[0]:].all() == np.eye(B.shape[0], dtype = int).all()

    M = np.shape(H)[0] # N-K
    N = np.shape(H)[1] 
    K = N - M
    G = np.concatenate([np.eye(K), ((-1)*B[:, :K].T %2)], axis=1).astype(bool)
    
    A = np.copy(H)
    for j in range(H.shape[1]):
        h = H.take(permut[j], axis = 1)
        A[:, j] = h[:]
            
    assert (G @ A.T % 2).all() == np.zeros((K, K)).all()
    return G, A


def MSA(H, y, SNR, max_iter = 10, threshold = 0.5):
    # SNR: Signal-Noise Ratio. SNR = 10 * lg(1 / variance) =  -20 lg (sigma) in dB.
    sigma = 10 ** (- SNR / 20)
    r = y * (2. / sigma ** 2)
    belief = r
    
    L = np.copy(H).astype(float)
    
    #initialization
    for i in range(L.shape[0]):
        L[i, :] = (L[i, :] > 0) * belief[:]

    for k in range(max_iter):
        #print(k)
        #row step
        for i in range(L.shape[0]):
            magnitudes = sorted(set(abs(L[i])), reverse = False)
            magnitudes.remove(0)
            min1, min2 = list(magnitudes)[:2]
            argmin1 = np.where(abs(L[i]) == min1)[0]
            signs = np.sign(L[i])
            parity = np.sum(signs < 0) % 2

            L[i] = (L[i] != 0) * min1
            L[i][argmin1] = min2
            L[i] *= signs * (-1) ** parity

        #column step
        belief = np.sum(L, axis = 0) + r
        L = (H != 0) * (belief - L)
        
        if (sum(H @ demodulate(belief) % 2) == 0):
            return belief
        if (np.prod(np.abs(belief) > threshold)):
            return belief
        
    return belief

def demodulate(b):
    u = 0.5 * (1 - np.sign(b))
    return u.astype(bool)

def modulate(v):
    return 1 - 2*v

def encoder(G, v, SNR, seed = 2021):
    # SNR: Signal-Noise Ratio. SNR = 10 * lg(1 / variance) =  -20 lg (sigma) in dB.
    # BPSK modulation: "0" -> +1., "1" -> -1.
    
    x = modulate(v)
    
    sigma = 10 ** (- SNR / 20)
    np.random.seed(seed = seed)
    e = awgn.rvs(size = x.shape, scale = sigma)
    
    y = x + e
    return y


def decoder(H, y, SNR, alg = 'MSA'):
    if alg == 'MSA':
        b = MSA(H, y, SNR = SNR)
    elif alg == 'SPA':
        b = SPA(H, y, SNR = SNR)
    else:
        print('unknown alg'); return np.zeros(y.shape).astype(bool)
    
    u = demodulate(b)
    
    s = H @ u % 2
    #if sum(s) != 0: print('probably there is an error')
    return u

@np.vectorize
def custom_llr(val):
    if abs(val) == 1.: return val
    return np.log((1 + val)/(1 - val))

def SPA(H, y, SNR, max_iter = 10, threshold = 0.5):
    # SNR: Signal-Noise Ratio. SNR = 10 * lg(1 / variance) =  -20 lg (sigma) in dB.
    sigma = 10 ** (- SNR / 20)
    r = y * (2. / sigma ** 2)
    
    M = np.copy(H).astype(float)
    E = np.zeros(H.shape)
    H_inv = (H + np.ones(np.shape(H))) %2

    for i in range(H.shape[0]):
        M[i, :] = r * H[i, :]

    for k in range(max_iter):
        #print(k)
        
        M = np.tanh(M/2) + H_inv
        t = np.prod(M, axis = 1) 
        for j in range(H.shape[1]):
            E[:, j] = t * H[:, j]
        E[:,:] = E[:, :] / M[:, :]
        E = H*E
        E = custom_llr(E)

        belief = np.sum(E, axis = 0) + r
        #print(belief)
        M = (H != 0) * (belief - E)

        if (sum(H @ demodulate(belief) % 2) == 0):
            return belief
        if (np.prod(np.abs(belief) > threshold)):
            return belief
        
    return belief

Z = 32
filler = np.vectorize(get_submatrix, otypes = [np.ndarray], signature = '()->(n,n)', excluded = (1,))
matrix = np.loadtxt('matrices/R1-1711982_BG2_set0.csv', dtype = int)
H = np.block([[x for x in row] for row in filler(matrix, Z)])

G, A = H2G(H)


with open('G.npy', 'wb') as f:
    np.save(f, G.astype(int))
with open('A.npy', 'wb') as f:
    np.save(f, A.astype(int))

with open('G.npy', 'rb') as f:
    G = np.load(f).astype(bool)
with open('A.npy', 'rb') as f:
    A = np.load(f).astype(bool)

print(A)
'''
#expirement
f = open("stats.txt", "a")
for SNR in np.arange(0., 3.01, 0.1):
    for n in range(5):
        print(SNR, n)
        a = np.random.randint(low = 0, high = 2, size = G.shape[0])
        v = (a @ G % 2).astype(bool)
        y = encoder(G = G, v = v, SNR = SNR, seed = n)
        u = decoder(H = A, y = y, SNR = SNR, alg = 'MSA')

        f.write("{:.2f}\t{}\t{:.6f}\n".format(SNR, int(np.allclose(u, v)), float(sum((u^v)))/len(u)))
f.close()
'''
