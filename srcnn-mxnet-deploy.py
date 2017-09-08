import os
import math
import scipy
import mxnet as mx
import numpy as np
from util import imread
from collections import namedtuple
Batch = namedtuple('label', ['data'])

def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 99.99
    MAX_PIXEL_VAL = 255
    return 20 * math.log10(MAX_PIXEL_VAL / math.sqrt(mse))

if __name__ == "__main__":
    magnitude = 2
    EpochNum  = 3
    nc = 1 # only luma channel

    ''' checkpoint '''
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    model_prefix = "checkpoint\\srcnn"

    ''' load test image '''
    test_image = os.path.join(os.getcwd(), 'Test\\Set5\\butterfly_GT.bmp')
    image = imread(test_image, is_grayscale=True)
    scipy.misc.imsave('ori_image.png', image)
    nh, nw = image.shape

    ''' generate bicubic interpolated images '''
    MAX_PIXEL_VAL = 128
    mod_input = (image - MAX_PIXEL_VAL) / MAX_PIXEL_VAL
    bicubic = scipy.ndimage.interpolation.zoom(mod_input, (1./magnitude), prefilter=False)
    bicubic = scipy.ndimage.interpolation.zoom(bicubic, magnitude, prefilter=False)
    bicubic = (bicubic * MAX_PIXEL_VAL) + MAX_PIXEL_VAL
    bicubic.astype(int)
    psnr_bicubic = '%.4f' % compute_psnr(image, bicubic)
    scipy.misc.imsave('bicubic_image_PSNR_' + str(psnr_bicubic) +  '.jpg', bicubic)

    ''' load checkpoint '''
    srcnn, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, EpochNum)
    mod = mx.mod.Module(symbol=srcnn, context=mx.cpu(), label_names='label')
    mod.bind(for_training=False,
             data_shapes=[('data', (1, nc, nh, nw))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    mod.forward(Batch([mx.nd.array(mod_input).reshape([1, nc, nh, nw])]))
    mod_output = mod.get_outputs()[0].asnumpy()
    mod_output = np.dot(mod_output, MAX_PIXEL_VAL) + MAX_PIXEL_VAL
    srcnn_output = np.squeeze(mod_output)
    srcnn_output.astype(int)
    np.clip(srcnn_output, 0, MAX_PIXEL_VAL)
    psnr_srcnn = '%.4f' % compute_psnr(image[6:255-5, 6:255-5], srcnn_output)
    scipy.misc.imsave('srcnn_image_PSNR_' + str(psnr_srcnn) +  '.jpg', srcnn_output)
