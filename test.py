from sklearn.externals import joblib
from skimage.feature import hog
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Path to Testing Directory')
parser.add_argument('--pos', help='Path to positive images')
parser.add_argument('--neg', help='Path to negative images')

args = parser.parse_args()

pos_img_dir = args.pos
neg_img_dir = args.neg

clf = joblib.load('final_svm.pkl')

total_pos_samples = 0
total_neg_samples = 0

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
    h, w, d = img.shape
    l, t = int((w - 64)/2), int((h - 128)/2)
    crop = img[t:t+128, l:l+64]
    return crop

def make_data(f_pos, f_neg):
    pos_features = []
    neg_features = []
    for imgfile in f_pos:
        cur_path = os.path.join(pos_img_dir, imgfile)
        img = cv2.imread(cur_path)
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        pos_features.append(features.tolist())

    for imgfile in f_neg:
        cur_path = os.path.join(neg_img_dir, imgfile)
        img = cv2.imread(cur_path)
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        neg_features.append(features.tolist())

    return pos_features, neg_features

pos_img_files, neg_img_files = read_filenames()
pos_features, neg_features = make_data(pos_img_files, neg_img_files)
pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)

tp= cv2.countNonZero(pos_result)
fn = pos_result.shape[0] - tp
fp = cv2.countNonZero(neg_result)
tn = neg_result.shape[0] - fp

precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)

f1 = 2*precision*recall / (precision + recall)

print ("Precision: ", precision)
print ("Recall: ", recall)
print ("F1 Score: ", f1)
