import numpy as np
import cv2


def rcnn_feature_stats(imdb, layer, rcnn_model):
    # prev_rng = seed_rand()
    # conf = rcnn_config('sub_dir', imdb.name)
    save_file = '%s/feature_stats_%s_layer_%d_%s.npy' % ('cache_dir', imdb.name, layer, rcnn_model.cache_name)

    image_ids = np.array(imdb.image_index)
    num_images = min(image_ids.size, 200)
    boxes_per_image = 200
    image_ids = image_ids[np.random.permutation(image_ids.size)[0:num_images]]
    ns = np.ndarray((image_ids, boxes_per_image))
    for i in xrange(image_ids):
        # d = D()
        d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, imdb.name, image_ids[i])
        X = d['feat'][np.random.permutation(d['feat'].shape[0])[0:min(boxes_per_image, d['feat'].shape[0])], :]
        X = rcnn_pool5_to_fcX(X, layer, rcnn_model)

        # ns = cat(1, ns, np.sqrt(np.sum(X ** 2, 1)))
        ns[i, :] = np.sqrt(np.sum(X ** 2, 1))

    mean_norm = np.mean(ns)
    std = np.std(ns)
    np.save(save_file, (mean_norm, std))
    # rng(prev_rng)


def rcnn_load_cached_pool5_features(cache_name, imdb_name, id):
    return np.load('./feat_cache/%s/%s/%s.npz' % (cache_name, imdb_name, id))['arr_0'].all()


def rcnn_pool5_to_fcX(feat, layer, rcnn_model):
    layers = ['fc6', 'fc7']
    if layer > 5:
        for i in xrange(0, layer - 6 + 1):
            feat = np.dot(feat, rcnn_model.net.params[layers[i]][0].data.T) + rcnn_model.net.params[layers[i]][1].data
            feat[feat < 0] = 0
    return feat


def rcnn_scale_features(f, feat_norm_mean):
    target_norm = 20
    f *= (target_norm / feat_norm_mean)


def rcnn_extract_features(im, boxes, rcnn_model, layer = 'pool5'):
    batches, batch_padding = rcnn_extract_regions(im, boxes, rcnn_model)
    batch_size = rcnn_model.cnn.batch_size
    feat_dim = -1
    feat = []
    curr = 0
    for i in xrange(batches.shape[0]):
        batch = batches[i]
        # batch[:, :, 0, :], batch[:, :, 2, :] = batch[:, :, 2, :].copy(), batch[:, :, 0, :].copy()
        batch = batch.swapaxes(0, 3).swapaxes(1, 2)
        for j in xrange(batch_size):
            rcnn_model.net.blobs['data'].data[j] = batch[j]
        f = rcnn_model.net.forward()[layer]
        if i == 0:
            feat_dim = f.size / batch_size
            feat = np.zeros((boxes.shape[0], feat_dim), dtype = float)

        f = np.reshape(f, (batch_size, feat_dim)).swapaxes(0, 1)

        if i == batches.shape[0] - 1:
            if batch_padding > 0:
                f = f[:, :-batch_padding]

        feat[curr:curr + f.shape[1], :] = f.T
        curr = curr + batch_size

    return feat


def rcnn_extract_regions(im, boxes, rcnn_model):
    # im[:, :, 0], im[:, :, 2] = im[:, :, 2].copy(), im[:, :, 0].copy()
    num_boxes = boxes.shape[0]
    batch_size = rcnn_model.cnn.batch_size
    num_batches = int(np.ceil(float(num_boxes) / batch_size))
    batch_padding = batch_size - num_boxes % batch_size
    if batch_padding == batch_size:
        batch_padding = 0

    crop_mode = rcnn_model.detectors.crop_mode
    image_mean = rcnn_model.cnn.image_mean
    crop_size = image_mean.shape[0]
    crop_padding = rcnn_model.detectors.crop_padding

    batches = np.ndarray((num_batches, crop_size, crop_size, 3, batch_size), dtype = float)

    for batch in xrange(num_batches):
        batch_start = batch * batch_size
        batch_end = min(num_boxes, batch_start + batch_size)
        ims = np.ndarray((crop_size, crop_size, 3, batch_size), dtype = float)
        for j in xrange(batch_start, batch_end):
            bbox = boxes[j, :]
            crop = rcnn_im_crop(im, bbox, crop_mode, crop_size, crop_padding, image_mean)
            ims[:, :, :, j - batch_start] = crop.swapaxes(0, 1)

        batches[batch] = ims
    return batches, batch_padding


def rcnn_im_crop(im, bbox, crop_mode, crop_size, padding, image_mean):
    pad_w = 0
    pad_h = 0
    crop_width = crop_size
    crop_height = crop_size
    if padding > 0:
        scale = float(crop_size) / (crop_size - padding * 2)
        half_height = float(bbox[3] - bbox[1] + 1) / 2
        half_width = float(bbox[2] - bbox[0] + 1) / 2
        center = [bbox[0] + half_width, bbox[1] + half_height]
        bbox = np.reshape(np.round(np.array([center, center]) + np.array([[-half_width, -half_height], [half_width, half_height]]) * scale), (1, 4))[0].astype('int')
        unclipped_height = bbox[3] - bbox[1] + 1
        unclipped_width = bbox[2] - bbox[0] + 1
        pad_x1, pad_y1 = max(0, - bbox[0]), max(0, - bbox[1])
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(im.shape[1], bbox[2])
        bbox[3] = min(im.shape[0], bbox[3])
        clipped_height = bbox[3] - bbox[1] + 1
        clipped_width = bbox[2] - bbox[0] + 1
        scale_x = float(crop_size) / unclipped_width
        scale_y = float(crop_size) / unclipped_height
        crop_width = int(np.round(clipped_width * scale_x))
        crop_height = int(np.round(clipped_height * scale_y))
        pad_x1 = int(np.round(pad_x1 * scale_x))
        pad_y1 = int(np.round(pad_y1 * scale_y))
        pad_h = pad_y1
        pad_w = pad_x1
        if pad_x1 + crop_width > crop_size:
            crop_width = crop_size - pad_x1

        if pad_y1 + crop_height > crop_size:
            crop_height = crop_size - pad_y1

    tmp = im[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1, :].astype('float')
    tmp = cv2.resize(tmp, (crop_width, crop_height)) - image_mean[pad_h:crop_height + pad_h, pad_w:crop_width + pad_w, :]
    window = np.zeros((crop_size, crop_size, 3))
    window[pad_h:crop_height + pad_h, pad_w: crop_width + pad_w] = tmp
    return window


def rcnn_get_all_feature(imdb, rcnn_model):
    box_per_image = 200
    image_ids = imdb.image_index
    X = np.ndarray((0, 9216))
    y = np.ndarray((0,))
    for i in xrange(3):
        d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, imdb.name, image_ids[i])
        idx = np.where(d['gt_classes'] != 0)[0]
        idx = np.concatenate((idx, np.random.choice(np.where(d['gt_classes'] == 0)[0], box_per_image - idx.size)))
        X = np.concatenate((X, d['feat'][idx, :]))
        y = np.concatenate((y, d['gt_classes'][idx]))
    X = rcnn_pool5_to_fcX(X, rcnn_model.opts.layer, rcnn_model)
    return X.astype('f'), y.astype('i')
