import json
import cPickle as pickle
import os
from pprint import pprint
import time

import cv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from settings import *
from logic import handle_buffer

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

    faces, labels = pickle.load(open(ROOT_DIR + 'memo'))

    '''
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

    with open(ROOT_DIR + 'memo', 'w') as f:
        pickle.dump((faces, labels), f)
    '''

    for filename in os.listdir(EXTRA_POSITIVE_TRAINING):
        image = cv.LoadImageM(EXTRA_POSITIVE_TRAINING + filename)
        found, grayscale = find_faces(image)
        if len(found) != 1:
            continue
        face = found[0][0]
        small = cv.GetSubRect(grayscale, face)
        fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
        cv.Resize(small, fixed_size)
        faces.append(np.asarray(fixed_size).flatten())
        labels.append(1.0)

    for directory in os.listdir(EXTRA_NEGATIVE_TRAINING):
        for filename in os.listdir(EXTRA_NEGATIVE_TRAINING + directory):
            image = cv.LoadImageM(EXTRA_NEGATIVE_TRAINING + '%s/%s' % (directory, filename))
            found, grayscale = find_faces(image)
            if len(found) != 1:
                continue
            face = found[0][0]
            small = cv.GetSubRect(grayscale, face)
            fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
            cv.Resize(small, fixed_size)
            faces.append(np.asarray(fixed_size).flatten())
            labels.append(-1.0)

    faces = np.array(faces)
    X_train, X_test, y_train, y_test = train_test_split(faces, np.array(labels), test_size=.3)

    n_components = 150
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    #clf = KNeighborsClassifier(weights='distance')
    clf = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)

    # now time to use the full set
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(faces)
    X_pca = pca.transform(faces)
    clf = SVC(**clf.best_params_).fit(X_pca, np.array(labels))
    #clf = KNeighborsClassifier(weights='distance').fit(X_pca, np.array(labels))

    return clf, pca

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
    CHECK_TIME = 15
    clf, pca = load()

    cam = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()
    buff = []
    while True:
        frame = cv.QueryFrame(cam)
        faces, grayscale = find_faces(frame)
        validity = {}
        for (x, y, w, h), _ in faces:
            face = (x, y, w, h)
            small = cv.GetSubRect(grayscale, face)
            fixed_size = cv.CreateMat(80, 80, cv.CV_8UC1)
            cv.Resize(small, fixed_size)
            vec = np.asarray(fixed_size).flatten()
            vec_pca = pca.transform(vec)
            prediction = clf.predict(vec_pca)[0]
            validity[(x, y, w, h)] = (prediction == 1.0)

            if prediction == 1.0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv.Rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)

        cv.Flip(frame, None, 1)
        cv.ShowImage('window', frame)
        cv.WaitKey(1)

        buff.append(validity)
        if time.time() - start > CHECK_TIME:
            handle_buffer(buff)
        start = time.time()
        buff = []

def get_training_data():
    cam = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cv.SetCaptureProperty(cam, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        frame = cv.QueryFrame(cam)
        faces, grayscale = find_faces(frame)
        if len(faces) == 1:
            cv.SaveImage(EXTRA_POSITIVE_TRAINING + str(time.time()) + '.jpg', frame)

        for (x, y, w, h), _ in faces:
            cv.Rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        cv.Flip(frame, None, 1)
        cv.ShowImage('window', frame)
        cv.WaitKey(1)


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
    #save(*train_model())
    run()
