import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def unpickle(file : str) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_label_names():
    data = unpickle("cifar-10/cifar-10-batches-py/batches.meta")
    labels = data[b"label_names"]
    return [s.decode("utf-8")  for s in labels]

'''
conversion of the images from 3072 vectors to 32x32x3 matrices
'''
def extract_images(data : np.array) -> list[np.array]:
    images = []
    for image in data:
        image = image.astype(np.uint8)
        red = image[0:1024].reshape((32,32))
        green = image[1024:2048].reshape((32,32))
        blue =  image[2048:3072].reshape((32,32))
        res_image = np.ones((3, 32, 32)).astype(np.uint8)
        res_image[0, :, :] = red
        res_image[1, :, :] = green
        res_image[2, :, :] = blue
        images.append(res_image)
    return images


'''
the downloadable dataset is composed by 6 batch of 10k images each
the function merges all the 6 batches together
'''
def load_data():
    # load the training set
    train_images = []
    train_labels = []
    for b in range(1,6):
        batch = unpickle(f"cifar-10/cifar-10-batches-py/data_batch_{b}")
        labels = batch[b"labels"]
        data = batch[b"data"]
        images = extract_images(data)
        train_labels += labels
        train_images += images
    train_images = np.array(train_images, dtype=np.uint8)
    train_labels = np.array(train_labels, dtype=np.uint8)

    # load test set
    batch = unpickle("cifar-10/cifar-10-batches-py/test_batch")
    labels = batch[b"labels"]
    data = batch[b"data"]
    images = extract_images(data)
    test_images = np.array(images)
    test_labels = np.array(labels, dtype=np.uint8)
    

    return train_images, train_labels, test_images, test_labels
        



x_train, y_train, x_test, y_test = load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(get_label_names())