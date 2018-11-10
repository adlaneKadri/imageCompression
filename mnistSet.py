from builtins import hasattr

from numpy.random.mtrand import RandomState
from sklearn import decomposition
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from time import time
import  data


#train_images,train_labels,test_images,test_labels, sizeImg = data.data1()
train_images,train_labels,test_images,test_labels, sizeImg = data.data2()

#q2, to display a matrix in a picture
def show(imageMatrix):
    plt.gray()
    plt.imshow(imageMatrix)
    plt.show()


# to show a set of image in one image : or to display a number
# #showChiffre(nombre de ligne , nombre de colonne, l'index du premier chiifre a lire dans la table traint_image)
def showChiffre(train_images, nLine, mColumn, imgIndex):
    initMatrix = np.zeros((nLine * sizeImg, mColumn * sizeImg))
    print(initMatrix.shape)
    i = 0
    while i < nLine:
        j = 0
        while j < mColumn:
            o = 0;
            for s in range(i * sizeImg, (i + 1) * sizeImg):
                p = 0;
                for r in range(j * sizeImg, (j + 1) * sizeImg):
                    initMatrix[s][r] = train_images[imgIndex][o][p]
                    p += 1
                o += 1
            imgIndex += 1
            j += 1
        i += 1
    show(initMatrix)
    return initMatrix

#showChiffre(train_images,2,4,10)
#show(train_images[7])
print(train_images.shape)
print(test_images.shape)
#q4
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (8, 8)
rng = RandomState(0)

digits = load_digits()
faces = digits.data
n_samples, n_features = digits.data.shape

# global centering
faces_centered = faces



print("Dataset consists of %d Image Number" % n_samples)

def plot_gallery(title, train_images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(train_images):
        #show(comp.reshape(image_shape))
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape(image_shape),
                   cmap=cmap)



estimators = [
    ('PCA',
     decomposition.PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True),
     True),

    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False),
]


plot_gallery("data naturel", faces_centered[:n_components])
#diplay("data naturel", faces_centered[:n_components])

# #############################################################################
def start():
    for name, estimator, center in estimators:
        print("Extracting the top %d %s..." % (n_components, name))
        t0 = time()
        data = faces
        estimator.fit(data[2:2+n_row*n_col])
        train_time = (time() - t0)
        print("done in %0.3fs" % train_time)
        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_
        else:
            components_ = estimator.components_

        #mat = getComponentsMatrix(components_[:n_components])
        #print(mat.size)

        plot_gallery('%s - Train time %.1fs' % (name, train_time),
                     components_[:n_components])


def getComponents(component_):
    matrix_ = np.zeros((n_row * sizeImg, n_col * sizeImg))
    i = 0
    while i < n_row:
        j = 0
        while j < n_col:
            o = 0;
            for s in range(i * sizeImg, (i + 1) * sizeImg):
                p = 0;
                for r in range(j * sizeImg, (j + 1) * sizeImg):
                    if ((p<n_col*n_row) and (o < n_col*n_row)):
                        matrix_[s][r] = component_[o][p]
                    p += 1
                o += 1
            j += 1
        i += 1
    #show(matrix_)
    return matrix_


#start()
#plt.show()

