import os

ROOT_DIR = os.environ['HOME'] + '/.facelock/'
CREDENTIALS = ROOT_DIR + 'credentials.json'
PICTURES = ROOT_DIR + 'pictures/'
META = ROOT_DIR + 'meta.json'
CASCADE_FILE = ROOT_DIR + 'haarcascade_frontalface_alt.xml'
CLASSIFIER = ROOT_DIR + 'classifier.pickle'
PCA_FILE = ROOT_DIR + 'pca.pickle'
EXTRA_POSITIVE_TRAINING = ROOT_DIR + 'extra_train_positive/'
EXTRA_NEGATIVE_TRAINING = ROOT_DIR + 'extra_train_negative/'
