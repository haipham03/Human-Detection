from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import svm
import random
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Data Training Path')
parser.add_argument('--pos', help='Path to positive images')
parser.add_argument('--neg', help='Path to negative images')

MAX_HARD_NEGATIVES = 20000

args = parser.parse_args()
pos_img_dir = args.pos
neg_img_dir = args.neg


def read_filenames():

    pos_imgs = []
    neg_imgs = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        pos_imgs.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        neg_imgs.extend(filenames)
        break

    return pos_imgs, neg_imgs


def crop_centre(img):
    h, w, _ = img.shape
    l, t = int((w - 64)/2), int((h - 128)/2)
    crop = img[t:t+128, l:l+64]
    return crop

def random_crops(img):
    h, w = img.shape
    h -= 128
    w -= 64
    if h < 0 or w < 0:
        return []

    windows = []

    for i in range(0, 10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y+128, x:x+64])

    return windows


def make_data(pos_files, neg_files):

    X = []
    Y = []

    for img_file in pos_files:
        cur_path = os.path.join(pos_img_dir, img_file)
        img = cv2.imread(cur_path)
        cropped = crop_centre(img)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        X.append(features)
        Y.append(1)

    for img_file in neg_files:
        cur_path = os.path.join(neg_img_dir, img_file)
        img = cv2.imread(cur_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crops = random_crops(gray_img)
        for crop in crops:
            features = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            X.append(features)
            Y.append(0)


    return X, Y


pos_img_files, neg_img_files = read_filenames()


X, Y = make_data(pos_img_files, neg_img_files)

X = np.array(X)
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=0)

clf1 = svm.LinearSVC(C=0.01, class_weight='balanced', verbose = 1)

clf1.fit(X, Y)

joblib.dump(clf1, 'person_pre-eliminary.pkl')

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0]-128, step_size[1]):
        for x in range(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def hard_negative_mine(f_neg, winSize, winStride):
    hard_negatives = []
    hard_negative_labels = []
    cnt = 0
    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            features = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            if (clf1.predict([features]) == 1):
                hard_negatives.append(features)
                hard_negative_labels.append(0)
                cnt += 1
            if (cnt == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

    return np.array(hard_negatives), np.array(hard_negative_labels)

winStride = (8, 8)
winSize = (64, 128)

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

hard_negatives = np.concatenate((hard_negatives, X), axis = 0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis = 0)

hard_negatives, hard_negative_labels = shuffle(hard_negatives, hard_negative_labels, random_state=0)

clf2 = svm.LinearSVC(C=0.01 , class_weight='balanced', verbose = 1)

clf2.fit(hard_negatives, hard_negative_labels)

joblib.dump(clf2, 'person_final.pkl')
