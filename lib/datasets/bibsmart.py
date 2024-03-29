# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import errno
from datasets.imdb import imdb
import numpy as np
from PIL import Image
import scipy.sparse
import cPickle
import uuid
import json
import re

#from bibsmart_eval import bibsmart_eval

class bibsmart(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'bib')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg','.JPG','.png']
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_groundtruth_bibsmart.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} groundtruth bibsmart loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_bibsmart_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote groundtruth bibsmart to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        #roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_bibsmart_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of Bibsmart.
        """
        filter_file_match = re.findall('([\w\d-]+)_sat(\w+)', index) # for images that look like 775654-1019-0021_sat0.jpg
        if (len(filter_file_match) > 0):
            index = filter_file_match[0][0]
            
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        # print 'Loading: {}'.format(filename)
        with open(filename) as f:
            objs = json.load(f)

        num_objs = len(objs['annotations'])

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Get Image name for later
        imagePath = objs['filename']
        imageName = re.findall('([^\\/]*\.\w+)$', imagePath)
        imageName = imageName[0]

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs['annotations']):
            # Make pixel indexes 0-based
            x1 = int(obj['x'])
            y1 = int(obj['y'])
            x2 = int(obj['x'])+int(obj['width'])
            y2 = int(obj['y'])+int(obj['height'])
            confidence = 1
            if 'confidence' in obj:
                confidence = float(obj['confidence'])
            if confidence > 0.98:
                #Issues with sloth annotation setting coordinates below zero
                if (x1 < 0):
                    x1 = 0
                if (y1 < 0):
                    y1 = 0
                # Issues with sloth annotations greater than image width/height
                imageName = os.path.join("/home/soda/workspace/py-faster-rcnn/data/training_images/formatted/bibsmart_devkit/data/Images", imageName)
                img = Image.open(imageName)
                width, height = img.size

                if (x2 >= width):
                    x2 = width - 1
                if (y2 >= height):
                    y2 = height - 1
                boxes[ix, :] = [x1, y1, x2, y2]
                cls = self._class_to_ind['bib']
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_bibsmart_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = self._get_bibsmart_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_bibsmart_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_bibsmart_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_bibsmart_results_file_template(self):
        # Bibsmart_devkit/results/comp4-44503_det_test_{%s}.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._devkit_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    # def _do_python_eval(self, output_dir = 'output'):
    #     annopath = os.path.join(
    #         self._data_path,
    #         'Annotations',
    #         '{:s}.txt')
    #     imagesetfile = os.path.join(
    #         self._data_path,
    #         'ImageSets',
    #         self._image_set + '.txt')
    #     cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    #     aps = []
    #     if not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(self._classes):
    #         if cls == '__background__':
    #             continue
    #         filename = self._get_bibsmart_results_file_template().format(cls)
    #         rec, prec, ap = bibsmart_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
    #         aps += [ap]
    #         print('AP for {} = {:.4f}'.format(cls, ap))
    #         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
    #             cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #     print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print('{:.3f}'.format(ap))
    #     print('{:.3f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')

