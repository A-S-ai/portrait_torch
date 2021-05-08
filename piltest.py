from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from os.path import splitext
mask = cv2.imread('00001_matte.png', cv2.IMREAD_UNCHANGED)
#mat = np.unique(mask)
mat = np.where(mask > 0, 255, 0)
print(mat.dtype)
mat = mat.astype(np.uint8)
print(mat.dtype)
cv2.imshow('1', mat)
cv2.waitKey(0)
mat = np.unique(mat)
print(mat)




