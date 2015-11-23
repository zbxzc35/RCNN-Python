import numpy as np


def rcnn_test(rcnn_model, imdb, suffix = ''):
    image_ids = imdb.image_index
    feat_opts = rcnn_model.training_opts
    num_classes = rcnn_model.classes.shape[0]

    aboxes = [[] for _ in xrange(num_classes)]
    box_inds = [[] for _ in xrange(num_classes)]
    for i in xrange(num_classes):
        aboxes[i] = [[] for _ in xrange(len(image_ids))]
        box_inds[i] = [[] for _ in xrange(len(image_ids))]
    # aboxes = np.ndarray((num_classes, len(image_ids)))
    # box_inds = np.ndarray((num_classes, len(image_ids)))

    max_per_set = (100000 / 2500) * len(image_ids)
    max_per_image = 100
    top_scores = [[] for _ in xrange(num_classes)]
    # top_scores = cell(num_classes, 1)
    thresh = np.ones((num_classes, 1)) * -np.inf
    box_counts = np.zeros((num_classes, 1))

    cnt = 0
    for i in xrange(len(image_ids)):
        cnt += 1
        d = rcnn_load_cached_pool5_featurs(feat_opts.cache_name, imdb.name, image_ids[i])
        d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model)
        d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean)
        zs = d.feat * rcnn_model.detectors.W + rcnn_model.detectors.B
        for j in xrange(num_classes):
            boxes = d.boxes
            z = zs[:, j]
            I = np.where(d.gt != 0 and z > thresh[j])
            boxes = boxes[I, :]
            scores = z[I]
            aboxes[j][i] = np.concatenate((boxes.astype('float'), scores.astype('float')), 1)
            ord = np.argsort(scores)[::-1]
            ord = ord[0:np.min(ord.shape[0], max_per_image)]
            aboxes[j][i] = aboxes[j][i][ord, :]
            box_inds[j][i] = I[ord]

            box_counts[j] = box_counts[j] + ord.shape[0]
            top_scores[j] = np.concatenate((top_scores[j], scores[ord]), 0)
            if box_counts[j] > max_per_set:
                top_scores[j, max_per_set + 1:] = []
                thresh[j] = top_scores[j][-1]

    for i in xrange(num_classes):
        for j in xrange(len(image_ids)):
            # todo: to check
            # if ~isempty(aboxes{i}{j})
            I = np.where(aboxes[i][j][:, -1] < thresh[j])
            aboxes[i][j][I, :] = []
            box_inds[i][j][I, :] = []

        save_file = "".join([cache_dir, rcnn_model.classes[i], '_boxes_', imdb.name, suffix])
        boxes = aboxes[i]
        inds = box_inds[i]
        np.savez(save_file, boxes = boxes, inds = inds)






































