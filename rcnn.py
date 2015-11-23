from rcnn_create_model import *
# from rcnn_feature_stats import *
from rcnn_feature import *
from rcnn_load_model import *
import numpy as np
import caffe
from datasets.factory import get_imdb
from sklearn.linear_model import LogisticRegression
import theanets
import climate


def rcnn_train(imdb):
    use_gpu = False
    opts = Opts()
    opts.net_def_file = './model-defs/rcnn_batch_256_output_fc7.prototxt'
    conf = [imdb.name, 1, 'cachedir', 'cachedir/' + imdb.name]

    rcnn_model = RcnnModel(opts.net_def_file, opts.net_file, opts.cache_name)
    # rcnn_load_model(rcnn_model, use_gpu)
    rcnn_model.detectors.crop_mode = opts.crop_mode
    rcnn_model.detectors.crop_padding = opts.crop_padding
    rcnn_model.classes = imdb.classes
    rcnn_model.opts = opts

    X, y = rcnn_get_all_feature(imdb, rcnn_model)
    np.savez('feat', X = X, y = y)
    classifier = LogisticRegression(class_weight = 'balanced', solver = 'lbfgs', multi_class = 'multinomial', verbose = 1, n_jobs = -1, max_iter = 1000)
    classifier.fit(X, y)

    # climate.enable_default_logging()
    net = theanets.Classifier(layers=[4096, 21])
    net.train((X, y), algo = 'sgd', learning_rate = 0.1, momentum = 0.9, save_every = 60.0, save_progress = 'net.{}.netsave', validate_every = 100)

    # opts.feat_norm_mean = rcnn_feature_stats(imdb, opts.layer, rcnn_model)
    # print 'average norm = %.3f\n' % feat_norm_mean
    #
    # X_pos, keys_pos = get_positive_pool5_features(imdb, opts)
    #
    # caches = []
    # for i in imdb.class_ids:
    #     X_pos[i] = rcnn_pool5_to_fcX(X_pos[i], opts.layer, rcnn_model)
    #     rcnn_scale_features(X_pos[i], opts.feat_norm_mean)
    #     caches.append(Cache(X_pos[i], keys_pos[i]))
    #
    # first_time = True
    # max_hard_epochs = 1
    # for hard_epoch in xrange(max_hard_epochs):
    #     for i in xrange(len(imdb.image_ids)):
    #         [X, keys] = sample_negative_features(first_time, rcnn_model, caches, imdb, i)




class Cache():
    def __init__(self, X_pos, keys_pos):
        self.X_pos = X_pos
        self.X_neg = []
        self.keys_pos = keys_pos
        self.keys_neg = []
        self.num_added = 0
        self.retrain_limit = 2000
        self.evict_thresh = -1.2
        self.hard_thresh = -1.0001
        self.pos_loss = []
        self.neg_loss = []
        self.reg_loss = []
        self.tot_loss = []


def sample_negative_features(first_time, rcnn_model, caches, imdb, ind):

    opts = rcnn_model.train_opts
    d = rcnn_load_cached_pool5_features(opts.cache_name, imdb.name, imdb.image_ids[ind])
    class_ids = imdb.class_ids
    # todo: wating for check
    # if is empty (d['feat'])
    d['feat'] = rcnn_pool5_to_fcX(d['feat'], opts.layer, rcnn_model)
    d['feat'] = rcnn_scale_features(d['feat'], opts.feat_norm_mean)

    neg_over_thresh = 0.3

    if first_time:
        for cls_id in class_ids:
            # todo:
            pass
    else:
        zs = np.dot(d['feat'],  rcnn_model.detectors.W) + rcnn_model.detectors.B
        for cls_id in class_ids:
            z = zs[:, cls_id]
            # todo:
            pass


def get_positive_pool5_features(imdb, opts):
    # X_pos = np.ndarray((max(imdb.class_ids), 1), dtype = float)
    # keys = np.ndarray((max(imdb.class_ids), 1), dtype = float)
    X_pos = []
    keys = []

    for i in xrange(imdb.image_ids):
        d = rcnn_load_cached_pool5_features(opts.cache_name, imdb.name, imdb.image_ids[i])
        for j in imdb.class_ids:
            if not X_pos[j]:
                X_pos.append([])
                keys.append([])
                # X_pos[j] = []
                # keys[j] = []
            sel = np.where(d.c == j)[0]
            if sel:
                X_pos.append(d.feat[sel, :])
                keys.append([i * np.ones(sel.shape[0]), sel])
                # X_pos[j] = cat(1, X_pos[j], d.feat[sel, :])
                # keys[j] = cat(1, keys[j], [i * np.ones(sel.shape[0]) sel])

        return X_pos, keys

if __name__ == "__main__":
    VOCdevkit = './datasets/VOCdevkit2007'
    imdb_train = get_imdb('voc_2007_trainval')
    rcnn_train(imdb_train)