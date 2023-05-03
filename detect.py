from sklearn.externals import joblib
from skimage.feature import hog
from nms import nms
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

parser = argparse.ArgumentParser(description='To read image name')

parser.add_argument('-i', "--image", help="Path to the input image", required=True)
parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.5, type=float)
parser.add_argument('-v', '--visualize', help="Whether to visualize sliding window", action="store_true")
parser.add_argument('-w', '--winstride', help="Pixels moved per step", default=8, type=int)
parser.add_argument('-n', '--nms_threshold', help="nms threshold", default=0.2, type=float)
args = vars(parser.parse_args())


def pushBBox(i, j, score, c, bbox):
    x = int(j * pow(scaleRatio, c))
    y = int(i * pow(scaleRatio, c))
    w = int(64 * pow(scaleRatio, c))
    h = int(128 * pow(scaleRatio, c))
    bbox.append((x, y, score, w, h))


clf = joblib.load("person_final.pkl")


orig = cv2.imread(args["image"])

img = orig.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


scaleRatio = args["downscale"]
inverse = 1.0/scaleRatio
winStride = (args["winstride"], args["winstride"])
winSize = (128, 64)

bbox = []

h, w = gray.shape
cnt = 0

while (h >= 128 and w >= 64):

    print(gray.shape)

    h, w = gray.shape
    i = 0
    while i < h - 128:
        j = 0
        while j < w - 64:

            n_window = gray[i:i+winSize[0], j:j+winSize[1]]
            features = hog(n_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt = True)

            result = clf.predict([features])

            if result[0] == 1:
                confidence_score = clf.decision_function([features])
                if confidence_score > 0.65 : 
                    pushBBox(i, j, confidence_score, cnt, bbox)

            if args["visualize"]:
                visual = gray.copy()
                cv2.rectangle(visual, (j, i), (j+winSize[1], i+winSize[0]), (0, 0, 255), 2)
                cv2.imshow("visual", visual)
                cv2.waitKey(1)


            j += winStride[0]

        i += winStride[1]

    gray = cv2.resize(gray, (int(w*inverse), int(h*inverse)), interpolation=cv2.INTER_AREA)
    cnt += 1

print(bbox)

final_bbox = nms(bbox, args["nms_threshold"])

# for (a, b, score, c, d) in bbox:
#     cv2.rectangle(orig, (a, b), (a+c, b+d), (0, 255, 0), 2)

# cv2.imwrite('./sample_images/before_nms_output.png', img)



for (a, b, score, c, d) in final_bbox:
    cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imwrite('./sample_images/output.png', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
