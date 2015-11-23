import numpy as np


class RcnnModel:
    def __init__(self, cnn_def_file, cnn_binary_file, cache_name = 'none'):
        cnn = CNN(cnn_def_file, cnn_binary_file)
        self.cnn = cnn
        self.cache_name = cache_name
        self.detectors = Detectors()
        self.net = None


class Opts():
    def __init__(self):
        self.svm_C = 1e-3
        self.bias_mult = 10
        self.pos_loss_weight = 2
        self.layer = 7
        self.crop_mode = 'warp'
        self.crop_padding = 16
        self.net_file = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k'
        self.cache_name = 'v1_finetune_voc_2007_trainval_iter_70k'


class CNN:
    def __init__(self, cnn_def_file, cnn_binary_file):
        self.def_file = cnn_def_file
        self.binary_file = cnn_binary_file
        self.batch_size = 256
        self.init_key = -1
        self.input_size = 227
        image_mean = np.load('./data/ilsvrc_2012_mean.npy')
        image_mean = image_mean.swapaxes(0, 2)
        # image_mean[:, :, 0], image_mean[:, :, 2] = image_mean[:, :, 2].copy(), image_mean[:, :, 0].copy()
        off = np.floor(float(image_mean.shape[0] - self.input_size) / 2) + 1
        # todo: reverse dim
        self.image_mean = image_mean[off-1:off + self.input_size - 1, off-1:off + self.input_size - 1, :]


class Detectors:
    def __init__(self):
        self.W = []
        self.B = []
        self.crop_mode = 'warp'
        self.crop_padding = 16
        self.nms_thresholds = []
