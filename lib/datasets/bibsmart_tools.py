import json
import re
import os
import csv
from ast import literal_eval
from PIL import Image
from collections import OrderedDict

# Bash Commands
# Copying a random number of files from a directory
# ls --ignore={'*patch*','*t.jpg'} | shuf -n 10 | xargs -I % cp % /home/soda/Desktop/variety_2000/

def create_annotation_from_json(file, outputFolder):
    with open(file) as data_file:
        data = json.load(data_file)
        for ix in xrange(len(data)):
            currentData = data[ix]
            filePath = currentData['filename']
            objs = re.findall('\w+.\/(\w+).\w{3}', filePath)
            filename = objs[0]
            outputFile = os.path.join(outputFolder, filename + '.txt')
            with open(outputFile, 'w') as outfile:
                json.dump(currentData, outfile)

def create_annotation_from_detection(detection_file, image_path, outputFolder, set):
    delimiter = ','

    with open(detection_file,'r') as det_f:
        data_iter = csv.reader(det_f,
                               delimiter = delimiter,
                               quotechar = '"')
        data = [data for data in data_iter]

    imageGroup = ''
    annotationsList = list()
    jsonObject = OrderedDict()

    for ix in xrange(len(data)):
        annotation = OrderedDict()
        row = data[ix]
        regex = re.findall('([\w\d-]+)-patch.*.(.\w{3})', row[0])[0]
        imageName = regex[0] + regex[1]

        # first iteration
        if (not imageGroup):
            imageGroup = imageName

        patchName = row[0]
        coords = row[1]
        sourceImageSize = row[2]
        confidence = float(row[3])
        coords = normalizeCoords(image_path, imageName, sourceImageSize, coords)

        annotation['patchName'] = patchName
        annotation['confidence'] = confidence
        annotation['x'] = coords[0]
        annotation['y'] = coords[1]
        annotation['width'] = coords[2] - coords[0]
        annotation['height'] = coords[3] - coords[1]
        annotationsList.append(annotation)

        if (ix >= (len(data) - 1)):
            jsonObject['filename'] = imageName
            jsonObject['set'] = set
            jsonObject['annotations'] = annotationsList
            saveJson(imageName, jsonObject, outputFolder)
        else:
            regex = re.findall('([\w\d-]+)-patch.*.(.\w{3})', data[ix + 1][0])[0]
            nextImageName = regex[0] + regex[1]

            # check and see if the current iteration needs to be saved
            if (nextImageName != imageGroup):
                jsonObject['filename'] = imageName
                jsonObject['set'] = set
                jsonObject['annotations'] = annotationsList
                saveJson(imageName, jsonObject, outputFolder)

                imageGroup = nextImageName
                annotationsList = list()
                jsonObject = {}

def saveJson(fileName, jsonObject, outputFolder):
    fileName = os.path.splitext(fileName)[0]
    outputFile = os.path.join(outputFolder, fileName + '.txt')
    with open(outputFile, 'w') as outfile:
        json.dump(jsonObject, outfile)
        print("Annotation file saved to {0}".format(fileName))

def normalizeCoords(image_path, imageName, sourceImageSize, coords):
    imagePath = os.path.join(image_path, imageName)
    img = Image.open(imagePath)
    destImageWidth, destImageHeight = img.size

    sourceImageSize = literal_eval(sourceImageSize)
    sourceImageWidth = sourceImageSize[1]
    sourceImageHeight = sourceImageSize[0]

    percentDiffHeight = float(destImageHeight) / sourceImageHeight
    percentDiffWidth = float(destImageWidth) / sourceImageWidth

    # - patchCoordinates is like [x1, y1, x2, y2]
    coords = literal_eval(coords)
    nX1 = coords[0] * percentDiffWidth
    nY1 = coords[1] * percentDiffHeight
    nX2 = coords[2] * percentDiffWidth
    nY2 = coords[3] * percentDiffHeight
    return [nX1, nY1, nX2, nY2]



if __name__ == '__main__':

    outputFolder = os.path.join("/home/soda/workspace/py-faster-rcnn", "output", "Annotations")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # files = ["/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race1.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race2.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race4.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race5.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race6.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race7.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race11.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race12.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race13.json",
    #          "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race14.json"]
    # for file in files:
    #     create_annotation_from_json(file, outputFolder)

    set = "variety-1000"
    imagePath = "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/{0}/Images".format(set)
    detectionFile = "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/{0}/detection_gpu0_soda-desktopFalse.csv".format(set)
    create_annotation_from_detection(detectionFile, imagePath, outputFolder, set)