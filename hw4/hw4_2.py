
import numpy as np
import cv2 as cv
import math
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv.warpAffine(img,M,(cols,rows))
    return shifted

# create array to store images, 784 bc 28x28
images = np.zeros((5,784))
# and correct values
correct_vals= np.zeros((5,5))
i =0
no = 0
for letter in ['A','B','C','D','E']:
    #read image in grayscale (1 dimension)
    gray = cv.imread("images/im_"+str(letter)+".png", 0)

    #print(f'original: {gray.shape}')

    #resize image
    gray_res = cv.resize(gray, (28,28))

    #print(f'resized: {gray_res.shape}')
    (thresh, gray_res) = cv.threshold(gray_res, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


    while np.sum(gray_res[0]) == 0 :
        gray_res = gray_res[1:]

    while np.sum(gray_res[:,0]) == 0:
        gray_res = np.delete(gray_res,0,1)

    while np.sum(gray_res[-1]) == 0:
        gray_res = gray_res[:-1]

    while np.sum(gray_res[:,-1]) == 0:
        gray_res = np.delete(gray_res,-1,1)

    rows,cols = gray_res.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
		# first cols than rows
        gray_res = cv.resize(gray_res, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
		# first cols than rows
        gray_res = cv.resize(gray_res, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray_res = np.lib.pad(gray_res,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray_res)
    shifted = shift(gray_res,shiftx,shifty)
    gray_res = shifted

    #save
    cv.imwrite("image_"+str(letter)+".png", gray_res)

    # flatten
    flatten = gray_res.flatten() / 255.0

    #print(flatten.shape)

    images[i] = flatten
    correct_val = np.zeros((5))
    correct_val[no] = 1
    correct_vals[i] = correct_val
    print(images[i])
    i += 1
    no +=1

print(images)
print(correct_vals)

