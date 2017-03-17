import json
import re
import os

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



if __name__ == '__main__':
    files = ["/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race1.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race2.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race4.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race5.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race6.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race7.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race11.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race12.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race13.json",
             "/home/soda/workspace/py-faster-rcnn/data/training_images/bibsmart/Race_Results/sloth_race14.json"]

    outputFolder = "Annotations"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in files:
        create_annotation_from_json(file, outputFolder)