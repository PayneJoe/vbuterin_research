## naive implementation of m31 and qm31
## cm31 = m31[X] / X^2 - alpha, where i^2 = alpha = -1
## qm31 = cm31[X] / X^2 - beta, where u^2 = beta = i + 2 

import numpy as np
M31 = modulus = 2**31-1

## reduction for add/sub where v < 2 * M31
def partial_reduce(v):
    # assert(not np.any(v >= 2 * M31))
    return np.array([e if e < M31 else (e - M31) for e in v], dtype=np.uint64)

## reduction for mul/div where v < M31^2
def reduce(v):
    # assert(not np.any(v >= M31 ** 2))
    hi = v >> 31
    lo = v & M31
    return partial_reduce(hi + lo)

def add(lft, rht):
    return partial_reduce(lft + rht)

def neg(v):
    return np.array([e if e == 0 else (M31 - e) for e in v], dtype=np.uint64)

def sub(lft, rht):
    return partial_reduce(lft + neg(rht))

def mul(lft, rht):
    return reduce(lft * rht)

## naive implementation of expoentiation
def pow(x, bits):
    assert(int(bits[-1]) == 1)
    n = len(bits)
    acc = np.copy(x)
    for i in range(1, n):
        acc = mul(acc, acc)
        if int(bits[i]) == 1:
            acc = mul(acc, x)
        elif int(bits[i]) == 0:
            pass
        else:
            print('Exceptional bits')
            raise
    return acc

## v * v^-1 = 1 = v^{M31 - 1}
def modinv(v):
    assert((v == 0).sum() == 0)
    return pow(v, bin(M31 - 2)[2:])

## [(a + b * i) + (c + d * i) * j]^{-1} 
## 1. t = (a + b * i)^2 - (c + d * i)^2
## 2. 1 / t 
## 3. t^{-1} * ((a + b * i) - (c + d * i) * j)
def modinv_ext(v):
    assert(v.shape[-1] == 4)
    aa, bb, cc, dd = mul(v[:, 0], v[:, 0]), mul(v[:, 1], v[:, 1]), mul(v[:, 2], v[:, 2]), mul(v[:, 3], v[:, 3])
    ab, cd = mul(v[:, 0], v[:, 1]), mul(v[:, 2], v[:, 3])
    t0 = sub(add(aa, mul(2, add(dd, cd))), add(bb, mul(2, cc)))
    t1 = sub(add(mul(2, ab), dd), add(mul(4, cd), cc))
    t = add(mul(t0, t0), mul(t1, t1))
    t_inv = modinv(t)
    r0 = mul(t_inv, t0)
    r1 = neg(mul(t_inv, t1))
    # print(r0, r1)
    e0 = sub(mul(v[:, 0], r0), mul(v[:, 1], r1))
    e1 = add(mul(v[:, 1], r0), mul(v[:, 0], r1))
    e2 = neg(sub(mul(v[:, 2], r0), mul(v[:, 3], r1)))
    e3 = neg(add(mul(v[:, 2], r1), mul(v[:, 3], r0)))

    return np.array([e0, e1, e2, e3], dtype=np.uint64).transpose()

def div(lft, rht):
    return reduce(lft * modinv(rht)) 

def div_ext(lft, rht):
    return mul_ext(lft, modinv_ext(rht))

def mul_ext(lft, rht):
    assert(lft.shape[-1] == 4)
    assert(rht.shape[-1] == 4)
    a0a1, a0b1, a0c1, a0d1 = mul(lft[:, 0], rht[:, 0]), mul(lft[:, 0], rht[:, 1]), mul(lft[:, 0], rht[:, 2]), mul(lft[:, 0], rht[:, 3])
    b0a1, b0b1, b0c1, b0d1 = mul(lft[:, 1], rht[:, 0]), mul(lft[:, 1], rht[:, 1]), mul(lft[:, 1], rht[:, 2]), mul(lft[:, 1], rht[:, 3])
    c0a1, c0b1, c0c1, c0d1 = mul(lft[:, 2], rht[:, 0]), mul(lft[:, 2], rht[:, 1]), mul(lft[:, 2], rht[:, 2]), mul(lft[:, 2], rht[:, 3])
    d0a1, d0b1, d0c1, d0d1 = mul(lft[:, 3], rht[:, 0]), mul(lft[:, 3], rht[:, 1]), mul(lft[:, 3], rht[:, 2]), mul(lft[:, 3], rht[:, 3])
    e0 = add(sub(sub(a0a1, b0b1), add(c0d1, d0c1)), mul(2, sub(c0c1, d0d1)))
    e1 = add(add(add(a0b1, b0a1), sub(c0c1, d0d1)), mul(2, add(c0d1, d0c1)))
    e2 = sub(add(a0c1, c0a1), add(b0d1, d0b1))
    e3 = add(add(a0d1, b0c1), add(c0b1, d0a1))

    return np.array([e0, e1, e2, e3], dtype=np.uint64).transpose()

def zeros(shape):
    return np.zeros(shape, dtype=np.uint64)

def array(x):
    return np.array(x, dtype=np.uint64)

def arange(*args):
    return np.arange(*args, dtype=np.uint64)

def append(*args):
    return np.concatenate((*args,))

def tobytes(x):
    return x.tobytes()

def eq(x, y):
    return np.array_equal(x % M31, y % M31)

def iszero(x):
    return not np.any(x % M31)

def isone(x):
    return (x == 1).sum() == len(x)

def isone_ext(x):
    assert(x.shape[-1] == 4)
    n = x.shape[0]
    return ((x[:, 0] == 1).sum() == n) and ((x[:, 1] == 0).sum() == n) and ((x[:, 2] == 0).sum() == n) and ((x[:, 3] == 0).sum() == n)

import random

## test for m31
x = 3 ** np.arange(10**7, dtype=np.uint64) % M31
for _ in range(10):
   # mul
   idx0 = random.randrange(0, 10 ** 7)
   idx1 = random.randrange(0, 10 ** 7)
   lft, rht = x[idx0: idx0 + 4], x[idx1: idx1 + 4]
   assert(np.array_equal(mul(lft, rht), (lft * rht) % M31))

   # inversion
   idx2 = random.randrange(0, 10 ** 7)
   a = x[idx2: idx2 + 4]
   assert(isone(mul(a, modinv(a))))

## test for qm31
x = 3 ** np.arange(4 * (10 ** 7), dtype= np.uint64).reshape(((10 ** 7), 4)) % M31
for _ in range(1):
    idx = random.randrange(0, 10 ** 7)
    a = x[idx: idx + 4, :]
    # a = np.array([[1565101536,  400337306, 1201011910, 1455552083]], dtype= np.uint64)

    ## mul
    r0 = np.copy(a)
    h0 = np.copy(a)
    logn = 5 
    n = (2 ** logn) - 1
    for _ in range(logn):
        r0 = mul_ext(r0, r0)
    for _ in range(n):
        h0 = mul_ext(h0, a)
    assert np.array_equal(r0, h0)

    ## inversion
    a_inv = modinv_ext(a)
    assert(isone_ext(mul_ext(a, a_inv)))