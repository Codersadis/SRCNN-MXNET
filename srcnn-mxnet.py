import os
import logging
import mxnet as mx
from util import generate_data, read_h5file

''''
Network Architecture for SRCNN
'''
def SRCNN_Architecture(img_lr, filter_num1, kernel_size1, filter_num2, kernel_size2, filter_num3, kernel_size3):
    net = mx.sym.Convolution(data=img_lr, kernel=(kernel_size1, kernel_size1), stride=(1, 1),
                             pad=(4, 4), num_filter=filter_num1, name='conv1')
    net = mx.sym.Activation(data=net, act_type='relu', name='relu1')
    net = mx.sym.Convolution(data=net, kernel=(kernel_size2, kernel_size2), stride=(1, 1),
                             pad=(0, 0), num_filter=filter_num2, name='conv2')
    net = mx.sym.Activation(data=net, act_type='relu', name='relu2')
    net = mx.sym.Convolution(data=net, kernel=(kernel_size3, kernel_size3), stride=(1, 1),
                             pad=(2, 2), num_filter=filter_num3, name='conv3')
    return net

''''
Params for training samples
'''
logging.getLogger().setLevel(logging.DEBUG)
batch_size = 16
image_size = 33
label_size = 33
stride     = 15
magnitude  = 2
EpochNum   = 30

''''
Generate the h5 files
'''
if not os.path.exists(os.path.join(os.getcwd(), 'train.h5')):
    generate_data(image_size, stride, label_size, magnitude, "Train")
if not os.path.exists(os.path.join(os.getcwd(), 'val.h5')):
    generate_data(image_size, stride*3, label_size, magnitude, "Val")

''''
Read data from h5 file and convert to NDArrayIter.
'''
train_data, train_label = read_h5file(os.path.join(os.getcwd(), 'train.h5'))
val_data, val_label = read_h5file(os.path.join(os.getcwd(), 'val.h5'))
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, data_name='data', label_name='label')
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)

''''
Set network symbols
  loss function: LinearRegressionOutput means euclideanloss
  
'''
img_lr = mx.sym.Variable('data')
img_label = mx.sym.Variable('label')
net = SRCNN_Architecture(img_lr, 64, 9, 32, 1, 1, 5)
srcnn = mx.sym.LinearRegressionOutput(data=net, label=img_label, name='srcnn')

''''
Record checkpoint
'''
if not os.path.isdir("checkpoint"):
    os.mkdir("checkpoint")
model_prefix = "checkpoint\\srcnn"
checkpoint = mx.callback.do_checkpoint(model_prefix, period=1)

'''
Training process
'''
srcnn_model = mx.mod.Module(symbol=srcnn,
                            context=mx.cpu(),
                            data_names=['data'],
                            label_names=['label'])
srcnn_model.fit(train_iter,
                num_epoch=EpochNum,
                eval_data=val_iter,
                optimizer='adadelta',
                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in',magnitude=2),
                optimizer_params={'learning_rate':0.0001},
                eval_metric='mse',
                batch_end_callback=mx.callback.Speedometer(batch_size, 10),
                epoch_end_callback=checkpoint)
