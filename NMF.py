import random
from builtins import hasattr
from numpy.random.mtrand import RandomState
from sklearn import decomposition
from sklearn.datasets import load_digits
import numpy as np
from time import time
import data
import display



train_images,train_labels,test_images,test_labels, sizeImg = data.data2()


# __init__ variables :
n_row, n_col = 1,4
n_components = n_row*n_col
image_shape = (8,8)
rng = RandomState(0)
digits = load_digits()
data = digits.data

#NMF Informations:
estimators = [
    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False)]

# random_ is just the number of the first image that can be read in the data
# i start from  the image number 56 , if i want to start from a random number => random_= random.randint(0, 100)
#random_ = 56
random_= random.randint(0, 100)

#To start  the NMF application
def nmfLunch():
    print("NMF's started ....." )
    name, estimator, center = estimators[0]

    t0 = time()
    estimator.fit(data[random_:random_+n_components])
    train_time = (time() - t0)

    print("done in %0.3fs" % train_time)

    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    return components_
nmfResult = nmfLunch()

#CST  la taille d'une image : digit = 8
sizeImg = int(str(np.sqrt(nmfResult.shape[1])).split('.')[0])

# Display the NMF result ( matrix )
def displayNMF(images, nLine, mColumn):
    initMatrix = np.zeros((nLine *  sizeImg, mColumn *  sizeImg))
    i = 0
    index = 0
    while i < nLine:
        j = 0
        while j < mColumn:
            p = 0;
            for s in range(i * sizeImg, (i + 1) * sizeImg):
                #print(index)
                for r in range(j * sizeImg, (j + 1) * sizeImg):
                    initMatrix[s][r] = images[index][p]
                    p += 1
            index += 1
            j += 1
        i += 1
    display.show(initMatrix)
    return initMatrix


# return the W and H decopositions matrix of X ( data )
def NMF_WH_DECOMPOSITION(data):
    X= data
    from sklearn.decomposition import NMF
    model = NMF(n_components=2, init='random', random_state=0)
    random_ = 56
    X =data[random_:random_+n_components]
    W = model.fit_transform(X)
    H = model.components_
    return W

data_NMF = displayNMF(nmfResult,n_row,n_col)









