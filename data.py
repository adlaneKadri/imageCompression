import mnist
from sklearn.datasets import load_digits

#mnist dataset
def data1():
    # read mnist dataset
    train_images = mnist.get_train_images()
    train_labels = mnist.get_train_labels()

    test_images = mnist.get_test_images()
    test_labels = mnist.get_test_labels()

    #colormaps = cmap = "gray"
    sizeImg = 28
    return train_images,train_labels,test_images,test_labels,sizeImg

#sklearn dataset
def data2():
    digits = load_digits()
    # (1797,64 : images number , image's size

    train_images = digits.images
    train_labels = digits.target

    test_labels = digits.target_names
    test_images = digits.data

    sizeImg = 8
    return train_images,train_labels,test_images,test_labels,sizeImg
