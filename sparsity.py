
def sparsity(matrix):
    i = matrix.shape[0]
    j = matrix.shape[1]
    n = i*j
    sum = 0

    for ii in range(0,i):
        for jj in range(0,j):
            if matrix[ii].reshape(j,1)[jj] < 0.0001 :
                sum +=1

    return sum/n


