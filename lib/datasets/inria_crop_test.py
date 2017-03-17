import os
import numpy as np
import scipy.sparse
from PIL import Image

def _load_inria_annotation(path):
    """
    Load image and bounding boxes info from txt files of INRIAPerson.
    """
    num_classes = 2
    _classes = ('__background__', # always index 0
                         'person')
    _class_to_ind = dict(zip(_classes, xrange(num_classes)))

    files = os.listdir(os.path.join(path, 'data/Annotations'))
    files = sorted(files)
    for filename in files:
        filePath = os.path.join(path, 'data/Annotations', filename)
        print 'Loading: {}'.format(filePath)
        with open(filePath) as f:
            data = f.read()
        import re
        objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)

        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            coor = re.findall('\d+', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2])
            y2 = float(coor[3])
            cls = _class_to_ind['person']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        imageName = filename.replace(".txt", ".png")
        imagePath = os.path.join(path, 'data/Images', imageName)
        img = Image.open(imagePath)
        for ix in xrange(len(boxes)):
            print("Croppimg image: {0}) {1} at x1: {2} y1: {3} x2: {4} y2: {5}".format(ix, imageName, boxes[ix][0], boxes[ix][1], boxes[ix][2], boxes[ix][3]))
            crop = img.crop((boxes[ix][0], boxes[ix][1], boxes[ix][2], boxes[ix][3]))
            newFileName = imageName.replace(".png", str(ix) + ".png")
            crop.save(os.path.join(path, 'data/CropTest', newFileName))
        # return {'boxes' : boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps' : overlaps,
        #         'flipped' : False,
        #         'seg_areas' : seg_areas}

if __name__ == '__main__':
    path = "/home/soda/workspace/py-faster-rcnn/data/training_images/formatted/INRIA_Person_devkit"
    _load_inria_annotation(path)
