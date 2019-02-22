import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from builtins import hasattr
from sklearn import decomposition
import random

# (1797,64) : images number , image's size
digits = sklearn.datasets.load_digits()
(input_train,target_train),(input_test,target_test) = (digits.images,digits.target),(digits.data,digits.target)

xInput = tf.keras.utils.normalize(input_train, axis=1)
xTest = tf.keras.utils.normalize(input_test, axis=1)
xTest= xTest.reshape(1797,8,8)

"""
plt.imshow(xTest[100],cmap=plt.cm.binary)
plt.show()


plt.imshow(xInput[100],cmap=plt.cm.binary)
plt.show()

print(xInput[100])
print(xTest[100])"""


# create the model RNN :
nnModel = tf.keras.models.Sequential()
nnModel.add(tf.keras.layers.Flatten())

nnModel.add(tf.keras.layers.Dense(128,activation=tf.nn.relu,name='dense_1'))
nnModel.add(tf.keras.layers.Dense(128,activation=tf.nn.relu,name='dense_2'))
nnModel.add(tf.keras.layers.Dense(64,activation=tf.nn.relu,name='dense_3'))
# in the last layer , we have to use 10 neurone ( 0 -> 9 ]  with the boolean -activation- function
nnModel.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

nnModel.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy'
                ,metrics=['accuracy'] )


#start training :
nnModel.fit(xInput,target_train, epochs=120)


#calculate the validation loss and the validation accurancy :
validation_loss , validation_accuracy = nnModel.evaluate(xTest,target_test)
print("\n validation accuracy : ",validation_accuracy)
print("validation loss : ",validation_loss)



#test it :
test = nnModel.predict([xTest])

#print("prediction result is : ", np.argmax(test[1023]))
#plt.imshow(xTest[1023], cmap=plt.cm.binary)
#plt.show()





def nmfLunch(n_row,n_col,starting_point):
    n_components = n_row*n_col
    random_ = starting_point
    # NMF Informations:
    estimators = [
        ('Non-negative components - NMF',
         decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
         False)]
    print("\n\nNMF's started ....." )
    name, estimator, center = estimators[0]

    #NMF starting:
    estimator.fit(digits.data[random_:random_+n_components])
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    return components_

random_1= random.randint(0, 100)
random_2= random.randint(1, 20)

nmfResult = nmfLunch(1,random_2,random_1)

plt.imshow(nmfResult[0].reshape(8,8),cmap=plt.cm.binary)
plt.show()

xTest = tf.keras.utils.normalize(nmfResult, axis=1)
xTest = xTest.reshape(random_2,8,8)
test = nnModel.predict([xTest])

print("prediction result is : ", np.argmax(test[0]))
plt.imshow(xTest[0], cmap=plt.cm.binary)
plt.show()
#print(nmfResult[random_1])



# to display a matrix in a picture
def show(imageMatrix):
    plt.gray()
    plt.imshow(imageMatrix)
    plt.show()


sizeImg = 8
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
    show(initMatrix)

displayNMF(nmfResult,6,10)
