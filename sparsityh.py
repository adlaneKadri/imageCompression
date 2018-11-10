import numpy as np

def sparsityH(X):
    shape0 = X.shape[0]
    shape1 = X.shape[1]
    A = np.zeros((shape1,1))
    B = transfert(X)
    for i in range(0,(shape1-1)):
        if np.sqrt(np.sum(B[:, i]** 2)) ==0 :
            a2 = 0.00001
        else:
            a2= np.sqrt(np.sum(B[:, i] ** 2))

        #print(((np.sqrt(shape0) - (np.sum(np.abs(B[:, i])) / a2))/(np.sqrt(shape0) - 1))/ (np.sqrt(shape0) - 1) )
        A[i]=((np.sqrt(shape0) - (np.sum(np.abs(B[:, i])) / a2))/(np.sqrt(shape0) - 1))/ (np.sqrt(shape0) - 1)
    #print(np.mean(A))
    return (np.mean(A),A)



def transfert(X):
    shape0 = X.shape[0]
    shape1 = X.shape[1]

    A = np.zeros((shape0,shape1))
    for i in range(0,shape0):
        for j in range(0,shape1):
            A[i][j] = X[i].reshape(shape1,1)[j]
    return A
