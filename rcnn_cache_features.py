from datasets.factory import *
from rcnn_create_model import *
from rcnn_load_model import *
from rcnn_feature import *
import roi_data_layer.roidb as rdl_roidb
import cv2
import numpy as np
import os
import sys
import multiprocessing


def rcnn_cache_features(chunk):

    net_file = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k'
    cache_name = 'v1_finetune_voc_2007_trainval_iter_70k'
    crop_mode = 'warp'
    crop_padding = 16
    imdb_train = get_imdb('voc_2007_train')
    imdb_val = get_imdb('voc_2007_val')
    imdb_test = get_imdb('voc_2007_test')
    imdb_trainval = get_imdb('voc_2007_trainval')

    if chunk == 'train':
        rcnn_cache_pool5_features(imdb_train, crop_mode, crop_padding, net_file, cache_name)
        link_tranval(cache_name, imdb_train, imdb_trainval)
    elif chunk == 'val':
        rcnn_cache_pool5_features(imdb_val, crop_mode, crop_padding, net_file, cache_name)
        link_tranval(cache_name, imdb_val, imdb_trainval)
    elif chunk == 'test_1':
        end_at = int(np.ceil(len(imdb_test.image_index) / 2.))
        rcnn_cache_pool5_features(imdb_test, crop_mode, crop_padding, net_file, cache_name, end = end_at)
    elif chunk == 'test_2':
        start_at = int(np.ceil(len(imdb_test.image_index) / 2.)) + 1
        rcnn_cache_pool5_features(imdb_test, crop_mode, crop_padding, net_file, cache_name, start = start_at)


def link_tranval(cache_name, imdb_split, imdb_trainval):
    cmd = "".join(['mkdir -p ./feat_cache/', cache_name, '/', imdb_trainval.name, '; ',
                  'cd ./feat_cache/', cache_name, '/', imdb_trainval.name, '/; ',
                  'for i in `ls -1 ../', imdb_split.name, '`; ',
                  'do ln -s ../', imdb_split.name, '/$i $i; ', 'done;'])
    os.system(cmd)


def rcnn_cache_pool5_features(imdb, crop_mode, crop_padding, net_file, cache_name, start = 0, end = 0):
    opts = Opts()
    opts.net_def_file = './model-defs/rcnn_batch_256_output_pool5.prototxt'
    opts.output_dir = "".join(['./feat_cache/', cache_name, '/', imdb.name, '/'])
    rdl_roidb.prepare_roidb(imdb)
    roidb = imdb.roidb
    rcnn_model = RcnnModel(opts.net_def_file, net_file)
    # rcnn_load_model(rcnn_model)
    rcnn_model.detectors.crop_mode = crop_mode
    rcnn_model.detectors.crop_padding = crop_padding
    image_ids = imdb.image_index
    if end == 0:
        end = len(image_ids)
    pool = multiprocessing.Pool(12)
    for i in xrange(start, end):
        pool.apply_async(multi_wrapper, (opts, image_ids[i], roidb[i], imdb.image_path_at(i), i, rcnn_model))
    pool.close()
    pool.join()


def multi_wrapper(opts, image_id, roidb, path, i, rcnn_model):
    rcnn_load_model(rcnn_model)
    save_file = "".join([opts.output_dir, image_id, '.npz'])
    d = roidb
    im = cv2.imread(path)
    d['feat'] = rcnn_extract_features(im, d['boxes'], rcnn_model)
    np.savez_compressed(save_file, d)
    print 'feat %d done\n' % i


if __name__ == "__main__":
    rcnn_cache_features(sys.argv[1])