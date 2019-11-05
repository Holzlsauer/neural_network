"""
MNIST image loader
code by Holzlsauer
based on:
https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""
import os
import numpy as np
import struct as st
from imageio import imwrite


files = {
         'train_images': 'train-images-idx3-ubyte',
         'train_labels': 'train-labels-idx1-ubyte',
         'test_images': 't10k-images-idx3-ubyte',
         'test_labels': 't10k-labels-idx1-ubyte'
        }

def train_images():    
    """Convert the binary images file to numpy array. :tp: train, test"""
    with open(os.pardir + '/dataset/' + files['train_images'], 'rb') as train:
        train.seek(0)
        magic = st.unpack('>4B', train.read(4))
        number_img = st.unpack('>I', train.read(4))[0] # number of images
        number_row = st.unpack('>I', train.read(4))[0] # number of rows
        number_column = st.unpack('>I', train.read(4))[0] # number of columns
        size = number_img * number_row * number_column # 1 total of bytes
        img_array = np.zeros((number_img, number_row, number_column))
        img_array = np.asarray(st.unpack('>' + 'B' * size, train.read(size)))
        img_array = img_array/255
        img_array = img_array.reshape((number_img, number_row, number_column))
    return img_array

def test_images():    
    """Convert the binary images file to numpy array. :tp: train, test"""
    with open(os.pardir + '/dataset/' + files['test_images'], 'rb') as train:
        train.seek(0)
        magic = st.unpack('>4B', train.read(4))
        number_img = st.unpack('>I', train.read(4))[0] # number of images
        number_row = st.unpack('>I', train.read(4))[0] # number of rows
        number_column = st.unpack('>I', train.read(4))[0] # number of columns
        size = number_img * number_row * number_column # 1 total of bytes
        img_array = np.zeros((number_img, number_row, number_column))
        img_array = np.asarray(st.unpack('>' + 'B' * size, train.read(size)))
        img_array = img_array/255
        img_array = img_array.reshape((number_img, number_row, number_column))
    return img_array

def train_labels():
    """Open the IDX labels file. :tp: train, test"""
    with open(os.pardir + '/dataset/' + files['train_labels'], 'rb') as labels:
        magic, size = st.unpack('>II', labels.read(8))
        lbl_array = np.asarray(st.unpack('>' + 'B' * size, labels.read(size)))
    return lbl_array

def test_labels():
    """Open the IDX labels file. :tp: train, test"""
    with open(os.pardir + '/dataset/' + files['test_labels'], 'rb') as labels:
        magic, size = st.unpack('>II', labels.read(8))
        lbl_array = np.asarray(st.unpack('>' + 'B' * size, labels.read(size)))
    return lbl_array

def save_img(np_array, name):
    """Save in jpeg the image from the pixel values in np array"""
    try:
        os.mkdir('images')
    except:
        pass
    imwrite(f'{os.pardir}/images/{name}.jpeg', np_array.astype(np.uint8))

if __name__ == '__main__':
    pass