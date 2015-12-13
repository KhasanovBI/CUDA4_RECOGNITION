import cv2
import os

CAFFE_ROOT_PATH = '../../dev/caffe/'
MODEL_ROOT_PATH = os.path.join(CAFFE_ROOT_PATH, 'models/bvlc_googlenet/')
MODEL_FILE = os.path.join(MODEL_ROOT_PATH, 'deploy.prototxt')
TRAINED_FILE = os.path.join(MODEL_ROOT_PATH, 'bvlc_googlenet.caffemodel')
MEAN_FILE = os.path.join(CAFFE_ROOT_PATH, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
LABEL_FILE = os.path.join(CAFFE_ROOT_PATH, 'data/ilsvrc12/synset_words.txt')

# Image's settings
OPACITY = 0.2
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)
THICKNESS = 1
