import random
from builtins import hasattr
from numpy.random.mtrand import RandomState
from sklearn import decomposition
from sklearn.datasets import load_digits
import numpy as np
import data
import display


#init of variables :
n_row, n_col = 1,4
n_components = n_row*n_col
image_shape = (8,8)
sizeImg = image_shape[0]
rng = RandomState(0)
digits = load_digits()
data = digits.data

# update2(X,W) :  a function that aims to update the matrix W by applying a certain formula on W and X
# W = W.*((2*X*X'*W)./(W*W'*X*X'*W + X*X'*W*W'*W));
def update2(X, W):
    A0 = np.dot(matrixCST(2, X.shape[0], X.shape[0]), X)
    A1 = np.dot(A0, X.T)
    A2 = np.dot(A1, W)

    B1 = np.dot(W, W.T)
    B2 = np.dot(B1, X)
    B3 = np.dot(B2, X.T)
    B4 = np.dot(B3, W)

    C1 = np.dot(X, X.T)
    C2 = np.dot(C1, W)
    C3 = np.dot(C2, W.T)
    C4 = np.dot(C3, W)

    D1 = np.add(B4, C4)
    D2 = np.divide(A2, D1)
    D3 = np.multiply(W, D2)
    W = D3
    return W

# update1() :  it is similar as the function 'update2()'
def update1(V, W):
    # VV^T calculated ahead of time
    VV = V * V.T
    num = VV * W
    denom = (W * (W.T * VV * W)) + (VV * W * (W.T * W))
    # W = np.multiply(W, np.divide(num, denom))

    W = np.divide(np.multiply(W, num), denom)

    # W = W .* (XX*W) ./ (W*(W'*XX*W) + XX*W*(W'*W));
    # W = W ./ norm(W);
    # normalize W TODO: check if L2 norm working similar to MATLAB
    W /= np.linalg.norm(W, 2)

    return W


#  matrixCST(cst, i, j) -->   matrixCST(4 , 2 , 3) =>  [[4,4,4][4,4,4]]
def matrixCST(cst, i, j):
    maxi = np.zeros((i, j))
    for aa in range(0, i):
        for bb in range(0, j):
            maxi[aa][bb] = cst
    return maxi


# this function has as input a list of data that represents set of images (set of images),
# and it returns a single image (paste the images on nLine and mColumn)
def dataTomatix(data, nLine, mColumn):
    initMatrix = np.zeros((nLine * sizeImg, mColumn * sizeImg))
    i = 0
    index = 0
    while i < nLine:
        j = 0
        while j < mColumn:
            p = 0;
            for s in range(i * sizeImg, (i + 1) * sizeImg):
                for r in range(j * sizeImg, (j + 1) * sizeImg):
                    initMatrix[s][r] = data[index][p]
                    p += 1
            index += 1
            j += 1
        i += 1
    return initMatrix


# X: data
# n_iter :iteration number for the W search
# k_ : the column number of the matrix W
# this function displays the result and also returns it
def projectiveNmf(X, n_iter, k):
    display.show(X)
    V = np.matrix(X)

    mi = np.zeros((X.shape[0], 1))
    W = np.matrix(np.random.rand(X.shape[0], k))

    for i in range(0, n_iter):
        W = update1(V, W)

    display.show(np.dot(np.dot(W, np.transpose(W)), X))
    #return W
    return np.dot(np.dot(W, np.transpose(W)), X)


random_= random.randint(0, 100)

k_ = 20
n_iter = 100
projDATA = dataTomatix(data[random_:random_+n_components], n_row, n_col)


#TEST
w = projectiveNmf(projDATA, k_, n_iter)
