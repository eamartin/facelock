import json
from pprint import pprint

import cv
import numpy as np

from settings import *

def find_faces(image):
    grayscale = cv.CreateImage(cv.GetSize(image), 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    storage = cv.CreateMemStorage()
    cv.EqualizeHist(grayscale, grayscale)

    cascade = cv.Load(CASCADE_FILE)
    faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (20, 20))
    return faces

def match_faces_and_tags(image_size, faces, tags):
    # map from faces to tags
    matches = {}
    pos_to_label = {}
    positions = []

    for tag in tags:
        pos = (int(image_size[0] * tag['pos'][0] / 100.0), int(image_size[1] * tag['pos'][1] / 100.0))
        pos_to_label[pos] = tag['type']
        positions.append(pos)

    for (x, y, width, height), _ in faces:
        for pos in positions:
            if x <= pos[0] <= x + width and y <= pos[1] <= y + height:
                face = (x, y, width, height)
                if face in matches:
                    print 'wtf'
                matches[face] = pos_to_label[pos]
    return matches

def train_model():
    w = cv.NamedWindow('window')
    with open(META) as f:
        meta = json.load(f)
    for entry in meta:
        # little reorganization
        tags = []
        me = {'type': 'me', 'pos': entry['tags']['me']}
        tags.append(me)
        for tag in entry['tags']['other']:
            tag = {'type': 'other', 'pos': tag}
            tags.append(tag)

        image = cv.LoadImageM(entry['picture'])
        found = find_faces(image)
        matches = match_faces_and_tags(cv.GetSize(image), found, tags)
        for (x, y, w, h), label in matches.iteritems():
            color = (255, 0, 0) if label == 'me' else (0, 255, 0)
            cv.Rectangle(image, (x, y), (x+w, y+h), cv.RGB(*color))
            small = cv.GetSubRect(image, (x, y, w, h))
            print (w, h)

        cv.ShowImage('window', image)
        cv.WaitKey(0)


if __name__ == '__main__':
    train_model()
