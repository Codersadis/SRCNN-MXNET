import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np
import random

'''
Read HDF5 file
Args:
    path[in]  : file path of h5 file
    data[out] : '.h5' file that contains data value
    label[out]: '.h5' file that contains label value
'''
def read_h5file(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = np.asarray(data)
        label = np.asarray(label)
        return data, label

'''
Preprocess data-set
Read image as gray-scale, then normalize and down-sample. 
Args:
    path[in]   : file path
    input_[out]: bicubic upsampled images [lr] 
    label_[out]: original images [hr]
'''
def preprocess(path, magnitude=2):
    image  = imread(path, is_grayscale=True)
    label_ = modcrop(image, magnitude)

    image  = image / 255
    label_ = label_ / 255

    input_ = scipy.ndimage.interpolation.zoom(label_, (1./magnitude), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, magnitude, prefilter=False)

    return input_, label_

'''
To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

We need to find modulo of height (and width) and scale factor.
Then, subtract the modulo from height (and width) of original image size.
There would be no remainder even after scaling operation.
'''
def modcrop(image, scale=2):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

"""
dataset: choose train dataset or validation set or test set
For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
Args:
  dataset[in]: image dataset
  data[out]  : image dataset absolute path
"""
def get_dataset_dir(datatype="Train"):
    if (datatype == "Train"):
        data = glob.glob(os.getcwd() + "//Train//*.bmp")
        return data
    elif (datatype == "Val"): # val set
        data = glob.glob(os.getcwd() + "//Test//Set14//*.bmp")
        return data

'''
Read image using its path.
Default value is gray-scale, and image is read by YCbCr format as the paper said.
'''
def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

'''
Generate data as h5 file format
Args:
  data[in]    : 4-D array data    
  label[in]   : 4-D array label
  datatype[in]: Train/Val/Test
'''
def write_h5file(data, label, datatype="Train"):
    if (datatype == "Train"):
        savepath = os.getcwd() + '\\train.h5'
    elif (datatype == "Val"):
        savepath = os.getcwd() + '\\val.h5'

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

"""
Read image files and make their sub-images and saved them as a h5 file format.
"""
def generate_data(image_size, stride, label_size, magnitude, datatype):
    if (datatype is "Train"):
        data = get_dataset_dir(datatype="Train")
    elif (datatype is "Val"):
        data = get_dataset_dir(datatype="Val")

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2  # 6

    for i in range(len(data)):
        input_, label_ = preprocess(data[i], magnitude)

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        for x in range(0, h - image_size + 1, stride):
            for y in range(0, w - image_size + 1, stride):
                sub_input = input_[x:int(x + image_size), y:int(y + image_size)]  # [33 x 33]
                sub_label = label_[int(x + padding):int(x + padding + label_size),
                            int(y + padding):int(y + padding + label_size)]  # [21 x 21]

                # Make channel value
                sub_input = sub_input.reshape([1, image_size, image_size])
                sub_label = sub_label.reshape([1, label_size, label_size])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [batch, ch, 33, 33]
    arrlabel = np.asarray(sub_label_sequence)  # [batch, ch, 21, 21]

    # shuffle the data, shuffle when loading into MXNET NDArrayIter
    shuffle_tmp = list(zip(arrdata, arrlabel))
    random.shuffle( shuffle_tmp )
    arrdata, arrlabel = zip(*shuffle_tmp)

    write_h5file(arrdata, arrlabel, datatype=datatype)
    write_h5file(arrdata, arrlabel, datatype=datatype)

