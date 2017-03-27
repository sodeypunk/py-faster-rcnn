#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from collections import OrderedDict
from natsort import natsorted
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, gc
import argparse
import re, csv

CLASSES = ('__background__',
           'bib')

NETS = {'bibsmart': ('bibsmart',
                  'bibsmart_faster_rcnn_final.caffemodel')}

class patch_info:
    def __init__(self):
        self.patch_name = ''
        self.image_data = None
        self.detection_coordinate = []
        self.ensemble_label_length = {}
        self.ensemble_label = {}
        self.ensemble_score = {}
        self.group_key = 0
        self.best_label = '-1'
        self.best_label_percent = 0
        self.best_coordinate = []
        self.final_labels_score = []

def apply_brightness_contrast_filters(image):
    FILTERS_PHI = [0.9]
    FILTERS_THETA = [1]
    dark_image = None
    light_image = None
    for f_num in xrange( len(FILTERS_PHI) ):
        phi = FILTERS_PHI[f_num]
        theta = FILTERS_THETA[f_num]
        maxIntensity = 255.0  # depends on dtype of image data
        for intensity_num in xrange( 2 ):
            # Apply filters for both intensities
            if intensity_num == 0:
                # Create Hi intensity image
                print("Creating lightened image")
                light_image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 0.5
            else:
                # Create Low intensity image
                print("Creating darkened image")
                dark_image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2

    return light_image, dark_image




def vis_detections(im, class_name, patch_info_list, image_name, output_path, detection_confidence, recon_confidence):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    groups = {}

    for ix, patch in enumerate(patch_info_list):
        key = patch.group_key
        if (key in groups) == False:
            groups[key] = list()
            groups[key].append(patch)
        else:
            groups[key].append(patch)

    for key in groups:
        bbox = groups[key][0].best_coordinate
        label = groups[key][0].best_label
        score = groups[key][0].best_label_percent

        if ('-1' in label):
            continue

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{0} - {1}'.format(label, round(score,2)),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{0} detections with '
                  'p({0} | box) >= detection {1} and recognition {2}').format(class_name,
                                                  detection_confidence, recon_confidence),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    plt.savefig(os.path.join(output_path, image_name))
    plt.close()

def expand_patch(image_size, coordinate, percent):
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    width = x2 - x1
    height = y2 - y1

    width_increase = (width * percent) - width
    height_increase = (height * percent) - height

    new_x1 = x1 - (width_increase / 2)
    new_y1 = y1 - (height_increase / 2)
    new_x2 = x2 + (width_increase / 2)
    new_y2 = y2 + (height_increase / 2)

    if (new_x1 < 0):
        new_x1 = 0
    if (new_y1 < 0):
        new_y1 = 0
    if (new_x2 > image_size[1]):
        new_x2 = image_size[1]
    if (new_y2 > image_size[0]):
        new_y2 = image_size[0]

    return [new_x1, new_y1, new_x2, new_y2]

def extract_patch(image_name, ordered_images_dict, total_dets, patch_size, thresh):
    increase_coordinate_percent = 1.2
    indexes = np.where(total_dets[:, 4] >= thresh)[0]
    if len(indexes) == 0:
        return

    filtered_coordinates = total_dets[indexes, :]

    # Expand all coordinates by 20%
    for ix, coordinate in enumerate(filtered_coordinates):
        filtered_coordinates[ix][:4] = expand_patch(ordered_images_dict['normal'].shape, coordinate, increase_coordinate_percent)

    # Get NMS groups
    coordinate_groups_dict = non_max_suppression(filtered_coordinates, 0.2)

    image_name_no_ext = re.findall('([^\\/]*)\.\w+$', image_name)[0]
    patch_info_list = list()

    for groupKey in coordinate_groups_dict:
        groupIndexes = coordinate_groups_dict[groupKey]
        coordinates = filtered_coordinates[groupIndexes, :]

        for ix, coordinate in enumerate(coordinates):
            new_patch = patch_info()
            new_patch.group_key = groupKey
            x1 = coordinate[0]
            y1 = coordinate[1]
            x2 = coordinate[2]
            y2 = coordinate[3]
            image_type_index = int(coordinate[5])
            image = ordered_images_dict.items()[image_type_index][1]
            image_type = ordered_images_dict.items()[image_type_index][0]
            new_patch_image = image[y1:y2, x1:x2]
            new_patch_image = resize_img(new_patch_image, patch_size[1], patch_size[0])
            new_patch.patch_name = '{0}-{1}-patch-{2}-{3}'.format(image_name_no_ext, image_type, groupKey,ix)
            new_patch.image_data = new_patch_image
            new_patch.detection_coordinate = coordinate
            patch_info_list.append(new_patch)
    return patch_info_list

def resize_img(img, width, height):
    iw = int(width)
    ih = int(height)
    # img = img.resize((iw, ih), Image.ANTIALIAS)
    img = cv2.resize(img, (iw, ih))
    return img

def save_patches(patches):
    patch_path = os.path.join(output_path, "patches")
    if os.path.exists(patch_path) == False:
        os.makedirs(patch_path)
    for i, patch in enumerate(patches):
        patch_name = os.path.join(patch_path, patch.patch_name + ".jpg")
        cv2.imwrite(patch_name, patch.image_data)

def GCN(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).

    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.

    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.

    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.

    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.

    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.

    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].

    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

def NormalizeData(imageArray):
    tempImageArray = imageArray

    # Normalize the data in batches
    batchSize = 25000

    dataSize = tempImageArray.shape[0]
    imageChannels = tempImageArray.shape[1]
    imageHeight = tempImageArray.shape[2]
    imageWidth = tempImageArray.shape[3]

    for i in xrange(0, dataSize, batchSize):
        stop = i + batchSize
        print("Normalizing data [{0} to {1}]...".format(i, stop))
        dataTemp = tempImageArray[i:stop]
        dataTemp = dataTemp.reshape(dataTemp.shape[0], imageChannels * imageHeight * imageWidth)
        # print("Performing GCN [{0} to {1}]...".format(i, stop))
        dataTemp = GCN(dataTemp)
        # print("Reshaping data again [{0} to {1}]...".format(i, stop))
        dataTemp = dataTemp.reshape(dataTemp.shape[0], imageChannels, imageHeight, imageWidth)
        # print("Updating data with new values [{0} to {1}]...".format(i, stop))
        tempImageArray[i:stop] = dataTemp
    del dataTemp
    gc.collect()

    return tempImageArray

def image_list_to_numpy_array(image_list, patch_size):

    num_images = len(image_list)
    total_image_array = np.zeros(shape=(num_images, 1, patch_size[0], patch_size[1]))

    for i, image in enumerate(image_list):
        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array[np.newaxis, ...]
        total_image_array[i] = image_array

    return total_image_array

def non_max_suppression(coordBoxes, overlapThresh):

    if len(coordBoxes) == 0:
        return []

    x1 = coordBoxes[:, 0]
    y1 = coordBoxes[:, 1]
    x2 = coordBoxes[:, 2]
    y2 = coordBoxes[:, 3]
    scores = coordBoxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    #indexes = np.argsort(y2)
    indexes = scores.argsort()[::-1]
    origSortedIndexes = indexes.copy()

    indexDict = {}
    while indexes.size > 0:
        #last = len(indexes) - 1
        i = indexes[0]

        # find the largest (x, y) coordinates for the start of
        #  the bounding box and the smallest (x, y) coordinates
        #  for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[indexes[0:]])
        yy1 = np.maximum(y1[i], y1[indexes[0:]])
        xx2 = np.minimum(x2[i], x2[indexes[0:]])
        yy2 = np.minimum(y2[i], y2[indexes[0:]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        inter = w * h
        overlap = inter / (area[i] + area[indexes[0:]] - inter)

        # boxes to suppress
        #suppress = np.concatenate(([i],
        #    np.where(overlap > overlapThresh)[0]))
        suppress = np.where(overlap > overlapThresh)[0]
        indexDict[i] = indexes[suppress]

        inds = np.where(overlap <= overlapThresh)[0]
        indexes = indexes[inds]

        # delete all indexes from the index list that are in the
		# suppression list
        #indexes = np.delete(indexes, suppress)



    return indexDict

class label_stat:
    def __init__(self):
        self.count = 0
        self.percent_of_total = 0
        self.total_label_count = 0

def combine_labels(current_label, current_label_score, new_label, new_label_score, min_score):

    if (new_label_score > min_score):
        # Condition 1
        if (new_label_score > current_label_score):
            return True

        # Condition 2
        if ((current_label in new_label) and (abs(new_label_score - current_label_score) < .02)):
            return True

    return False

def find_best_label_from_ensemble_group(patch_info_group, confidence_threshold):
    labels_dict = {}
    best_coordinate = []
    total_stat_count = 0

    ensemble_length = len(patch_info_group[0].ensemble_label)
    for ix, patch in enumerate(patch_info_group):
        for ensemble_index in xrange(ensemble_length):

            total_stat_count = total_stat_count + 1

            # Find all lengths and labels above confidence
            length_score = patch.ensemble_score[ensemble_index][0]
            if (length_score > confidence_threshold):
                # If length score is correct, make sure all labels for that length are also over threshold
                label_length = patch.ensemble_label_length[ensemble_index]

                label = ''
                for x in xrange(label_length):
                    label_score = patch.ensemble_score[ensemble_index][x + 1]
                    if (label_score > confidence_threshold):
                        ensemble_label = patch.ensemble_label[ensemble_index]
                        label_at_pos = ensemble_label[x]
                        label = label + str(label_at_pos)
                    else:
                        label = '-1'
                        break;

                if (label != '-1'):
                    if (label in labels_dict) == False:
                        new_label_stat = label_stat()
                        new_label_stat.count = 1
                        labels_dict[label] = new_label_stat
                    else:
                        labels_dict[label].count = labels_dict[label].count + 1

                    best_coordinate = patch.detection_coordinate

    # Build the final scores for length and label
    final_labels_score = {}

    for key in labels_dict:
        labels_dict[key].total_label_count = total_stat_count
        occurence_count = labels_dict[key].count
        percentOfTotal = float(occurence_count) / total_stat_count
        labels_dict[key].percent_of_total = percentOfTotal

        final_labels_score[key] = percentOfTotal

    best_label = '-1'
    best_label_percent = 0
    best_label_percent_min = 0.02 # minimum percentage
    if (len(final_labels_score) > 0):
        for key in final_labels_score:
            if (combine_labels(best_label, best_label_percent, key, final_labels_score[key], best_label_percent_min)):
                best_label_percent = final_labels_score[key]
                best_label = key

    if (best_label != '-1'):
        print ("Label: {0}  Percent of total at each position greater than confidence of {1}: [{2}]"
               .format(best_label, confidence_threshold, str(best_label_percent)))
    return best_label, best_label_percent, best_coordinate, final_labels_score

def find_best_label(ensemble_index, patch_info_list, recognition_confidence_threshold):
    groups = {}

    for ix, patch in enumerate(patch_info_list):

        key = patch.group_key

        if (key in groups) == False:
            groups[key] = list()
            groups[key].append(patch)
        else:
            groups[key].append(patch)

    for key in groups:
        best_label, best_label_percent, best_coordinate, final_labels_score = find_best_label_from_ensemble_group(groups[key], recognition_confidence_threshold)

        for ix, patch in enumerate(groups[key]):
            patch.best_label = best_label
            patch.best_label_percent = best_label_percent
            patch.best_coordinate = best_coordinate
            patch.final_labels_score = final_labels_score

def do_recognition(ensemble_index, net, image_array, patch_info_list, batch_size, patch_size):
    dataSize = image_array.shape[0]
    patch_height = patch_size[0]
    patch_width = patch_size[1]

    for i in xrange(0, dataSize, batch_size):
        stop = i + batch_size
        if (stop > dataSize):
            stop = dataSize

        print('Loading patches for ensemble {0}: {1} to {2}'.format(ensemble_index, i, stop))
        data4D = image_array[i:stop]
        rows = data4D.shape[0]
        extraRows = np.zeros([batch_size, 1, patch_height, patch_width])
        extraRows[:rows] = data4D
        data4D = extraRows

        data4DLabels = np.zeros([batch_size, 1, 1, 1])
        net.set_input_arrays(data4D.astype(np.float32), data4DLabels.astype(np.float32))
        prediction = net.forward()

        lastIndex = stop - i
        for x in xrange(lastIndex):
            index1 = prediction['prediction1'][x].argmax()
            index2 = prediction['prediction2'][x].argmax()
            index3 = prediction['prediction3'][x].argmax()
            index4 = prediction['prediction4'][x].argmax()
            index5 = prediction['prediction5'][x].argmax()
            index6 = prediction['prediction6'][x].argmax()

            score1 = prediction['prediction1'][x][index1]
            score2 = prediction['prediction2'][x][index2]
            score3 = prediction['prediction3'][x][index3]
            score4 = prediction['prediction4'][x][index4]
            score5 = prediction['prediction5'][x][index5]
            score6 = prediction['prediction6'][x][index6]

            strLength = index1 + 1
            strLabel = str(index2) + str(index3) + str(index4) + str(index5) + str(index6)
            #strLabel = strLabel[0:strLength] # Want to see the entire string, not partial for later
            scores = list()
            scores.append(score1)
            scores.append(score2)
            scores.append(score3)
            scores.append(score4)
            scores.append(score5)
            scores.append(score6)

            patch_info_list[i+x].ensemble_label_length[ensemble_index] = strLength
            patch_info_list[i+x].ensemble_label[ensemble_index] = strLabel
            patch_info_list[i+x].ensemble_score[ensemble_index] = scores

    return patch_info_list

def CreateCSVFile(csvData, csvFileName):
    print("Creating csv file...".format(csvFileName))
    with open(csvFileName, 'wb') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        for x in xrange(len(csvData)):
            writer.writerow(csvData[x])

def demo(net, recognition_net_list, image_name, im_folder, output_path):
    """Detect object classes in an image using pre-computed object proposals."""

    CONF_THRESH = 0.1
    RECON_CONF_THRESH = 0.98
    NMS_THRESH = 1.0
    PATCH_SIZE = [40, 60]

    # Load the demo image
    totalTimer = Timer()
    totalTimer.tic()
    images_dict = {}
    im_file = os.path.join(cfg.DATA_DIR, 'demo', im_folder, image_name)
    im_rgb = cv2.imread(im_file)
    im_light, im_dark = apply_brightness_contrast_filters(im_rgb)
    images_dict['normal'] = im_rgb
    images_dict['light'] = cv2.convertScaleAbs(im_light)
    images_dict['dark'] = cv2.convertScaleAbs(im_dark)
    ordered_images_dict = OrderedDict(sorted(images_dict.items()))

    # Save light and dark images  - Used for debugging only
    #cv2.imwrite(os.path.join(output_path, 'light.jpg'), im_light)
    #cv2.imwrite(os.path.join(output_path, 'dark.jpg'), im_dark)

    # Detect all object classes and regress object bounds
    patches_info_list = list()
    timer = Timer()
    timer.tic()
    total_dets = None
    images_type_count = 0
    for image_type_key in ordered_images_dict:
        im = images_dict[image_type_key]
        scores, boxes = im_detect(net, im)

        # Convert image to black and white using BGR2GRAY and not RGB2GRAY
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ordered_images_dict[image_type_key] = im
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            #add the images type to end of dets
            images_type_col = np.zeros(shape=(dets.shape[0], 1))
            images_type_col[:] = images_type_count
            dets = np.append(dets, images_type_col, 1)

            if (total_dets == None):
                total_dets = dets
            else:
                total_dets = np.append(total_dets, dets, 0)
            images_type_count = images_type_count + 1

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals for %d image').format(timer.total_time, boxes.shape[0], len(images_dict))

    # Perform recognition on patches
    timer = Timer()
    timer.tic()

    # Get all patches
    # dets = [[x1, y1, x2, y2, confidence, imageTypeIndex]]
    patches_info_list = extract_patch(image_name, ordered_images_dict, total_dets, PATCH_SIZE, CONF_THRESH)

    # Save patches - Used for debugging only
    save_patches(patches_info_list)

    # Convert images to numpy array
    image_list = list()
    for ix, patch in enumerate(patches_info_list):
        image_list.append(patch.image_data)
    numpy_patches = image_list_to_numpy_array(image_list, PATCH_SIZE)

    # Normalize patches
    normalized_patches = NormalizeData(numpy_patches)

    # Perform recognition on boxes
    batch_size = 50
    for ix, net in enumerate(recognition_net_list):
        do_recognition(ix, net, normalized_patches, patches_info_list, batch_size, PATCH_SIZE)

    # Find best labels for each patch group
    find_best_label(ix, patches_info_list, RECON_CONF_THRESH)

    timer.toc()
    print ('Recognition took {:.3f}s for '
       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    print('Recognition complete for image: {0}'.format(image_name))

    vis_detections(im_rgb, cls, patches_info_list, image_name, output_path, CONF_THRESH, RECON_CONF_THRESH)

    totalTimer.toc();
    print ('Total time took {:.3f}s for %s'.format(totalTimer.total_time, image_name))

    return patches_info_list

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [bibsmart]',
                        choices=NETS.keys(), default='bibsmart')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR,
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    recognition_net_conv3 = caffe.Net('/home/soda/workspace/bibsmart/models/recognition/deploy/set6/3conv/bibsmart_recognize_3conv.prototxt',
                                '/home/soda/workspace/bibsmart/models/recognition/deploy/set6/3conv/3conv_iter_150000.caffemodel',
                                caffe.TEST)
    recognition_net_conv4 = caffe.Net('/home/soda/workspace/bibsmart/models/recognition/deploy/set6/4conv/bibsmart_recognize_4conv.prototxt',
                                '/home/soda/workspace/bibsmart/models/recognition/deploy/set6/4conv/4conv_iter_150000.caffemodel',
                                caffe.TEST)
    recognition_net_conv5 = caffe.Net('/home/soda/workspace/bibsmart/models/recognition/deploy/set6/5conv/bibsmart_recognize_5conv.prototxt',
                                '/home/soda/workspace/bibsmart/models/recognition/deploy/set6/5conv/5conv_iter_150000.caffemodel',
                                caffe.TEST)
    recognition_net_conv6 = caffe.Net('/home/soda/workspace/bibsmart/models/recognition/deploy/set6/6conv/bibsmart_recognize_6conv.prototxt',
                                '/home/soda/workspace/bibsmart/models/recognition/deploy/set6/6conv/6conv_iter_175000.caffemodel',
                                caffe.TEST)
    recognition_net_conv7 = caffe.Net('/home/soda/workspace/bibsmart/models/recognition/deploy/set6/7conv/bibsmart_recognize_7conv.prototxt',
                                '/home/soda/workspace/bibsmart/models/recognition/deploy/set6/7conv/7conv_iter_150000.caffemodel',
                                caffe.TEST)
    recognition_net_list = [recognition_net_conv3, recognition_net_conv4, recognition_net_conv5, recognition_net_conv6, recognition_net_conv7]

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    test_set = "variety_test"
    path = os.path.join(cfg.DATA_DIR, 'demo', test_set)
    output_path = os.path.join('output', test_set)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    results_detailed_file_name = os.path.join(output_path, 'resultsDetailed.csv')
    result_list = list()
    im_names = sorted(os.listdir(path))
    total_all_images_timer = Timer()
    total_all_images_timer.tic()
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)

        # Run detection and recognition
        patches_info_list = demo(net, recognition_net_list, im_name, test_set, output_path)

        # Write out results to CSV file
        max_ensemble_size = len(recognition_net_list)
        for ix, patch in enumerate(patches_info_list):
            row_list = list()
            row_list.append(patch.patch_name)
            row_list.append(patch.best_label)
            row_list.append(patch.best_label_percent)
            row_list.append(patch.final_labels_score)
            ensemble_label_list = list()
            for x in xrange(max_ensemble_size):
                length = patch.ensemble_label_length[x]
                ensemble_label_list.append(patch.ensemble_label[x][:length])
            row_list.append(ensemble_label_list)
            for x in xrange(max_ensemble_size):
                row_list.append(patch.ensemble_score[x])
            result_list.append(row_list)

    total_all_images_timer.toc()
    print ('Total time for job: {0} seconds \nAverage time per image is {1} secs/img'.format(total_all_images_timer.total_time, round(total_all_images_timer.total_time / (len(im_names)), 2)))
    CreateCSVFile(result_list, results_detailed_file_name)

    plt.show()
