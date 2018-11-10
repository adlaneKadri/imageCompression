
import matplotlib.pyplot as plt
import numpy as np
import data

train_images,train_labels,test_images,test_labels, sizeImg = data.data2()




# to display a matrix in a picture
def show(imageMatrix):
    plt.gray()
    plt.imshow(imageMatrix)
    plt.show()


# to show a set of image in one image : or to display a number
# #showChiffre(nombre de ligne , nombre de colonne, l'index du premier chiifre a lire dans la table traint_image)
def showChiffre(train_images, nLine, mColumn, imgIndex):
    initMatrix = np.zeros((nLine * sizeImg, mColumn * sizeImg))
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
#mat = showChiffre(train_images,2,20,0)

n_row, n_col = 1,4
n_components = n_row*n_col
image_shape = (8,8)
data = test_images


def diplay(title, images, mColumn=n_col, nLine=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    initMatrix = np.zeros((nLine * 8, mColumn * 8))
    imgIndex = 0
    for i, comp in enumerate(train_images):
       print(comp.size)
       print(i)
    plt.show()


