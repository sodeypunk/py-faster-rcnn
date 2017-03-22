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
from collections import defaultdict
from natsort import natsorted
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, gc
import argparse
import re

CLASSES = ('__background__',
           'bib')

NETS = {'bibsmart': ('bibsmart',
                  'bibsmart_faster_rcnn_final.caffemodel')}

class patch_info:
    def __init__(self):
        self.patch_name = ''
        self.image_data = None
        self.detection_coordinate = []
        self.label = ''
        self.recognition_score = []
        self.group_key = 0
        self.best_label = ''
        self.best_coordinate = []
        self.label_ratio = 0

def vis_detections(im, class_name, dets, image_name, output_path, thresh=0.5):
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
    #plt.draw()
    plt.savefig(os.path.join(output_path, image_name))
    plt.close()

def extract_patch(image, image_name, norm_coordinates, patch_size, thresh):
    indexes = np.where(norm_coordinates[:, -1] >= thresh)[0]
    if len(indexes) == 0:
        return

    filtered_coordinates = norm_coordinates[indexes, :]

    # Get NMS groups
    coordinate_groups_dict = non_max_suppression(filtered_coordinates, 0.40)

    image_name_no_ext = re.findall('([^\\/]*)\.\w+$', image_name)[0]
    patch_info_list = list()

    for groupKey in coordinate_groups_dict:
        groupIndexes = coordinate_groups_dict[groupKey]
        coordinates = filtered_coordinates[groupIndexes, :]

        for ix, coordinate in enumerate(coordinates):
            new_patch = patch_info()
            new_patch.group_key = groupKey
            new_patch_image = image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
            new_patch_image = resize_img(new_patch_image, patch_size[1], patch_size[0])
            new_patch.patch_name = image_name_no_ext + "-patch-" + str(groupKey) + "-" + str(ix)
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

    pick = []
    x1 = coordBoxes[:, 0]
    y1 = coordBoxes[:, 1]
    x2 = coordBoxes[:, 2]
    y2 = coordBoxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(y2)
    origSortedIndexes = indexes.copy()

    indexDict = {}
    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)
        suppress = [last] # create array of size indexes
        #print("last is {0}".format(last))
        for pos in xrange(0, last):
            j = indexes[pos]
            #print("Comparing index {0} and {1}".format(i, j))
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinatestest
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            #print("overlap is: {0}.".format(overlap))

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                #print("Supressing pos {0} which is index {1}".format(pos, j))
                suppress.append(pos)

        indexDict[i] = origSortedIndexes[suppress]
        origSortedIndexes = np.delete(origSortedIndexes, suppress)
        #print ("index {0} overlaps with {1}".format(i, origSortedIndexes[suppress]))
        indexes = np.delete(indexes, suppress)
    return indexDict

def FindLabelFromGroup(patch_info_group, confidenceThreshold):
    bestLabel = -1
    bestAverageScore = 0
    bestCurrentArea = 0
    bestX1 = -1
    bestY1 = -1
    bestX2 = -1
    bestY2 = -1
    bestLengthScore = -1
    sameLabelDict = {}
    sameLabelDict[bestLabel] = 1
    newSortedList = list()
    labelRatio = -1

    for ix, patch in enumerate(patch_info_group):
        imageLabel = patch.label

        if (imageLabel in sameLabelDict):
            sameLabelDict[imageLabel] = sameLabelDict[imageLabel] + 1
        else:
            sameLabelDict[imageLabel] = 1

    # print("SameLabelDict: {0}".format(sameLabelDict))
    newSortedList = natsorted(patch_info_group, key=attrgetter('label'))
    # print("Dictionary: {0}".format(sameLabelDict))
    for ix, patch in enumerate(newSortedList):
        imageLabel = patch.label
        sumScores = 0
        goodResult = True
        for x in xrange(len(imageLabel) + 1):
            score = float(patch.recognition_score[x])
            if (score >= confidenceThreshold):
                sumScores = sumScores + score
            else:
                goodResult = False
                break  # break out of the current for loop

        if (goodResult == True):
            averageScore = float(sumScores / len(imageLabel))

            x1 = patch.detection_coordinate[1]
            y1 = patch.detection_coordinate[2]
            x2 = patch.detection_coordinate[3]
            y2 = patch.detection_coordinate[4]
            lengthScore = float(patch.detection_coordinate[0])
            currentArea = (x2 - x1 + 1) * (y2 - y1 + 1)
            sameLabelRatio = float(sameLabelDict[imageLabel]) / float(sameLabelDict[bestLabel])

            sameButHighScore = (imageLabel == bestLabel or bestLabel < 0) and averageScore > bestAverageScore
            sameLengthHigherScore = len(str(imageLabel)) == len(str(bestLabel)) and averageScore > bestAverageScore  # and currentArea <= bestCurrentArea
            greaterLengthScore = lengthScore > bestLengthScore and (str(imageLabel) not in str(bestLabel)) and (sameLabelRatio >= 1) and currentArea <= bestCurrentArea
            greaterLengthScoreAndLength = lengthScore > bestLengthScore and len(str(imageLabel)) > len(str(bestLabel))
            bestInsideCurrent = str(bestLabel) in str(imageLabel) and len(str(bestLabel)) < len(str(imageLabel)) and ((str(imageLabel).find(str(bestLabel)) == 0 and bestX2 < x2) or (str(imageLabel).find(str(bestLabel)) > 0 and bestX1 > x1))
            moreInstancesofCurrent = sameLabelRatio >= 2 and (str(imageLabel) not in str(bestLabel))
            greaterLengthAndPreviousLabelAtIndex0 = str(imageLabel).find(str(bestLabel)) == 0 and len(str(bestLabel)) < len(str(imageLabel)) and sameLabelRatio >= 2

            #resultTextList.append("Patch: {0},Label: {1},bestLabel: {2},lengthScore: {3},bestLengthScore: {4},area: {5},bestArea: {6},avgScore: {7},bestAvgScore: {8},sameLabelRatio: {9},sbhs: {10},slhs: {11},gls: {12},glsl: {13},mioc: {14},bic:{15},glli: {16}".format(imageFile, imageLabel, bestLabel, lengthScore, bestLengthScore, currentArea, bestCurrentArea, averageScore, bestAverageScore, sameLabelRatio, sameButHighScore, sameLengthHigherScore, greaterLengthScore, greaterLengthScoreAndLength, moreInstancesofCurrent, bestInsideCurrent, greaterLengthAndPreviousLabelAtIndex0))
            if ((sameButHighScore)
                or (sameLengthHigherScore)
                or (greaterLengthScore)
                or (greaterLengthScoreAndLength)
                # or moreInstancesofCurrent
                or (bestInsideCurrent)
                or (greaterLengthAndPreviousLabelAtIndex0)):


                bestCurrentArea = currentArea
                bestAverageScore = averageScore
                bestLengthScore = lengthScore
                bestX1 = x1
                bestY1 = y1
                bestX2 = x2
                bestY2 = y2
                bestLabel = imageLabel
                labelRatio = float(sameLabelDict[bestLabel]) / float(len(patch_info_group))

    return bestLabel, [bestX1, bestY1, bestX2, bestY2], labelRatio

def find_best_label(patch_info_list, recognition_confidence_threshold):
    key = -1
    groups = {}

    for ix, patch in enumerate(patch_info_list):
        if key == -1:
            key = patch.group_key

        if key in groups == False:
            groups[key] = list()
            groups[key].append(patch)
        else:
            groups[key].append(patch)

    for key in groups:
        label, best_coordinate, label_ratio = FindLabelFromGroup(groups[key], recognition_confidence_threshold)

        for ix, patch in enumerate(groups[key]):
            patch.best_label = label
            patch.best_coordinate = best_coordinate
            patch.label_ratio = label_ratio

def do_recognition(net, image_array, patch_info_list, batch_size, patch_size):
    dataSize = image_array.shape[0]
    patch_height = patch_size[0]
    patch_width = patch_size[1]

    for i in xrange(0, dataSize, batch_size):
        stop = i + batch_size
        if (stop > dataSize):
            stop = dataSize

        print('Loading patches {0} to {1}'.format(i, stop))
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
            strLabel = strLabel[0:strLength]
            scores = list()
            scores.append(score1)
            scores.append(score2)
            scores.append(score3)
            scores.append(score4)
            scores.append(score5)
            scores.append(score6)

            patch_info_list[i+x].label = strLabel
            patch_info_list[i+x].recognition_score = scores

    return patch_info_list

def demo(net, recognition_net_list, image_name, im_folder, output_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', im_folder, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Perform recognition on patches
    timer = Timer()
    timer.tic()
    # Read in image again as black and white because the conversion from RGB to grayscale darkened the image
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    CONF_THRESH = 0.1
    RECON_CONF_THRESH = 0.98
    NMS_THRESH = 1.1
    PATCH_SIZE = [40, 60]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        # Get all patches
        # dets = [[x1, y1, x2, y2, confidence][x1, y1, x2, y2, confidence]]
        patches_info_list = extract_patch(im, image_name, dets, PATCH_SIZE, CONF_THRESH)

        # Save patches
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
        do_recognition(recognition_net_list[0], normalized_patches, patches_info_list, batch_size, PATCH_SIZE)
        do_recognition(recognition_net_list[1], normalized_patches, patches_info_list, batch_size, PATCH_SIZE)
        do_recognition(recognition_net_list[2], normalized_patches, patches_info_list, batch_size, PATCH_SIZE)
        do_recognition(recognition_net_list[3], normalized_patches, patches_info_list, batch_size, PATCH_SIZE)
        do_recognition(recognition_net_list[4], normalized_patches, patches_info_list, batch_size, PATCH_SIZE)

        # Find best labels for each patch group
        find_best_label(patches_info_list, RECON_CONF_THRESH)

        timer.toc()
        print ('Recognition took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        print('Recognition complete for image: {0}'.format(image_name))
        #vis_detections(im, cls, dets, image_name, output_path, thresh=CONF_THRESH)

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

    test_set = "variety_100"
    path = os.path.join(cfg.DATA_DIR, 'demo', test_set)
    output_path = os.path.join('output', test_set)

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    im_names = os.listdir(path)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, recognition_net_list, im_name, test_set, output_path)

    plt.show()
