import cv2
import numpy as np
from datasets.factory import get_imdb
from rcnn_feature import *
from utils.nms import *
from rcnn_load_model import *
import theanets
from utils.selective_search import *
# import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os


def rcnn_demo(i):
    imdb = get_imdb('voc_2007_test')
    img = cv2.imread(imdb.image_path_at(i))
    rcnn_model = np.load('rcnn_model2.npy').all()
    rcnn_load_model(rcnn_model, False)
    # rcnn_model = np.load('rcnn_model.npz')['rcnn_model'].all()

    # thresh = -1
    dets = rcnn_detect(imdb.image_index[i], imdb, rcnn_model)
    for i in xrange(len(dets)):
        if isinstance(dets[i], list):
            continue
        vis_detections(img, rcnn_model.classes[i], dets[i])
    # all_dets = np.array([])
    # for i in xrange(len(dets)):
    #     if dets[i] is not []:
    #         all_dets = np.concatenate((all_dets, np.concatenate((i, dets[i]), 1)))
    #
    # ord = np.argsort(all_dets[:, -1])[::-1]
    # for i in xrange(ord.shape[0]):
    #     score = all_dets[ord[i], -1]
    #     if score < 0:
    #         break
    #     cls_name = rcnn_model.classes[all_dets[ord[i], 0]]
    #     vis_detections(img, cls_name, dets)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def rcnn_detect(imidx, imdb, rcnn_model):

    d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, imdb.name, imidx)
    d['feat'] = rcnn_pool5_to_fcX(d['feat'], rcnn_model.opts.layer, rcnn_model)
    # boxes = selective_search(img, ks = 500)
    # boxes = boxes.swapaxes(0, 1).swapaxes(2, 3)
    # feat = rcnn_extract_features(img, boxes, rcnn_model)
    # feat = rcnn_scale_features(feat, rcnn_model.training_opts.feat_norm_mean)
    scores = rcnn_model.classifier.predict_proba(d['feat'].astype('f'))
    # scores = feat * rcnn_model.detectors.W + rcnn_model.detectors.B

    scores_idx = np.argmax(scores, 1)
    num_classes = len(rcnn_model.classes)
    dets = [[] for _ in xrange(num_classes)]
    for i in xrange(1, num_classes):
        # I = np.where(scores[:, i] > thresh)
        I = np.where(scores_idx == i)[0]
        if I.size == 0:
            continue
        scored_boxes = np.concatenate((d['boxes'][I, :], scores[I, i].reshape((scores[I, i].size, 1))), 1)
        keep = nms(scored_boxes, 0.3)
        dets[i] = scored_boxes[keep, :]

    return dets

if __name__ == "__main__":
    rcnn_demo(12)
    rcnn_demo(15)
    os.system('sleep 1000')