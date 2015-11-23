import init
import caffe


def rcnn_load_model(rcnn_model, use_gpu = False, rcnn_model_file = None):
    if use_gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    # if rcnn_model_file is not None:
    #     rcnn_model.net = caffe.Net(rcnn_model_file, './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', caffe.TEST)
    # else:
    rcnn_model.net = caffe.Net(rcnn_model.cnn.def_file, rcnn_model.cnn.binary_file, caffe.TEST)
    # rcnn_model.cnn.layers = caffe('get_weights')
