from builtins import hasattr
from numpy.random.mtrand import RandomState
from sklearn import decomposition
from sklearn.datasets import load_digits
import numpy as np
from time import time
import data
import display
import random
import matplotlib.pyplot as plt
import scipy.misc

train_images,train_labels,test_images,test_labels, sizeImg = data.data2()



n_row, n_col = 1,4
n_components = n_row*n_col
image_shape = (8,8)
rng = RandomState(0)
digits = load_digits()
data = digits.data
estimators = [
    ('PCA',
     decomposition.PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True),
     True)]

random_ = random.randint(0, 100)

def plot_gallery(title, train_images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(train_images):
        #show(comp.reshape(image_shape))
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape(image_shape),
                   cmap=cmap)


def pcaLunch():
    print("NMF's started ....." )
    name, estimator, center = estimators[0]
    random_= random.randint(0, 100)
    #random_ = 31
    print(random_)
    t0 = time()
    estimator.fit(data[random_:random_+n_components])
    train_time = (time() - t0)

    print("done in %0.3fs" % train_time)

    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    return components_


pcaResult = pcaLunch()
print(pcaResult.shape)

#CST  la taille d'une image : digit = 8
sizeImg = int(str(np.sqrt(pcaResult.shape[1])).split('.')[0])

def displayPCA(images, nLine, mColumn):
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

dataPCA = displayPCA(pcaResult,n_row,n_col)











