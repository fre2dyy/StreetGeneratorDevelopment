
# Implements 8-connectivity connected component labeling
#
# Algorithm obtained from "Optimizing Two-Pass Connected-Component Labeling
# by Kesheng Wu, Ekow Otoo, and Kenji Suzuki
#

import cv2
import numpy as np


img = cv2.imread('files/motorway/gabor/motorway_gabor_S.tiff', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

connectivity = 4    # 4- OR 8-connectivity connected component labeling
retval, labels = cv2.connectedComponents(img, connectivity)


num = labels.max()

N = 50  # pixel threshold

# If the count of pixels less than a threshold, then set pixels to `0` (background)
for i in range(1, num+1):
    pts = np.where(labels == i)
    if len(pts[0]) < N:
        labels[pts] = 0

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

cv2.imwrite('files/motorway/gabor/motorway_gabor_S_ccl.tiff', labeled_img)
# np.savetxt("files/motorway/gabor/foo.txt", labels, fmt='%1.1d')