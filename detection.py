import datetime
import json
import cPickle as pickle
from pprint import pprint
import time

import cv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from settings import *

cv.NamedWindow('window')

def find_faces(image):
    grayscale = cv.CreateImage(cv.GetSize(image), 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    storage = cv.CreateMemStorage()
    cv.EqualizeHist(grayscale, grayscale)

    cascade = cv.Load(CASCADE_FILE)
    faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (20, 20))
    return faces, grayscale

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
                matches[face] = pos_to_label[pos]
    return matches

def train_model():
    with open(META) as f:
        meta = json.load(f)

    start = datetime.datetime.now()
    faces = []
    labels = []
    for entry in meta:
        # little reorganization
        tags = []
        me = {'type': 'me', 'pos': entry['tags']['me']}
        tags.append(me)
        for tag in entry['tags']['other']:
            tag = {'type': 'other', 'pos': tag}
            tags.append(tag)

        image = cv.LoadImageM(entry['picture'])
        found, grayscale = find_faces(image)
        matches = match_faces_and_tags(cv.GetSize(image), found, tags)
        for (x, y, w, h), label in matches.iteritems():
            if w >= 50 and h >= 50:
                small = cv.GetSubRect(grayscale, (x, y, w, h))
                fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
                cv.Resize(small, fixed_size)
                faces.append(np.asarray(fixed_size).flatten())
                labels.append(1.0 if label == 'me' else -1.0)

    faces = np.array(faces)
    X_train, X_test, y_train, y_test = train_test_split(faces, np.array(labels), test_size=.3)

    n_components = 150
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, 80, 80))

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    print y_pred
    print confusion_matrix(y_test, y_pred)
    return clf.best_estimator_, pca

def test(clf, pca):
    image = cv.LoadImageM(PICTURES + '10150462662796229.jpg')
    faces, grayscale = find_faces(image)
    results = []
    for (x, y, w, h), _ in faces:
        face = (x, y, w, h)
        small = cv.GetSubRect(grayscale, face)
        fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
        cv.Resize(small, fixed_size)

        vec = np.asarray(fixed_size).flatten()
        vec_pca = pca.transform(vec)
        results.append(clf.predict(vec_pca))
    print results


def run():
    clf, pca = load()

    cam = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        frame = cv.QueryFrame(cam)
        faces, grayscale = find_faces(frame)
        results = []
        for (x, y, w, h), _ in faces:
            face = (x, y, w, h)
            small = cv.GetSubRect(grayscale, face)
            fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
            cv.Resize(small, fixed_size)

            '''
            cv.ShowImage('window', fixed_size)
            cv.WaitKey()
            '''

            vec = np.asarray(fixed_size).flatten()
            vec_pca = pca.transform(vec)
            results.append(clf.predict(vec_pca)[0])
        print 1.0 in results
        time.sleep(5)

def save(clf, pca):
    with open(CLASSIFIER, 'w') as f:
        pickle.dump(clf, f)
    with open(PCA_FILE, 'w') as f:
        pickle.dump(pca, f)

def load():
    with open(CLASSIFIER) as f:
        clf = pickle.load(f)
    with open(PCA_FILE) as f:
        pca = pickle.load(f)
    return clf, pca


if __name__ == '__main__':
    run()
    '''
    clf, pca = train_model()
    print 'From memory'
    test(clf, pca)
    save(clf, pca)
    clf2, pca2 = load()
    print 'From disk'
    test(clf2, pca2)
    '''
