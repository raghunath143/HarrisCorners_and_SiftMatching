"""
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""
import foldrpp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    mask = np.zeros(R.shape)
    h = R.shape[0]
    w = R.shape[1]
    d_list = [[-1, -1], [-1, 0], [-1, +1],
              [0, -1], [0, +1],
              [+1, -1], [+1, 0], [+1, +1]]

    for i in range(0,int(R.shape[0])):
        for j in range(0,int(R.shape[1])):
            flag = True
            for d in d_list:
                ni, nj = (i + d[0], j + d[1])
                if (0 <= ni < h) and (0 <= nj < w):
                    if (R[ni, nj] >= R[i, j]):
                        flag = False
            if flag:
                mask[i,j] = 1
            else:
                mask[i,j] = 0

            # if(R[i-1,j-1] < R[i,j] and R[i,j-1] < R[i,j] and R[i-1,j] < R[i,j] and R[i+1,j] < R[i,j]
            # and R[i-1,j+1] < R[i,j] and R[i,j+1] < R[i,j] and R[i+1,j+1] < R[i,j] and R[i+1,j-1] < R[i,j]):
            #     mask[i,j] = 1
            # else:
            #     mask[i,j] = 0
    return mask



# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 29-30
# You can use opencv functions and numpy functions
def harris_corner(im):
    # step 0: convert RGB to gray-scale image
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray_1 = np.copy(gray)
    gray_1 = np.float32(gray_1)

    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    sobelx = cv2.Sobel(gray_1, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_1, cv2.CV_64F, 0, 1, ksize=5)

    # step 2: compute products of derivatives at every pixels
    IXX = np.square(sobelx)
    IYY = np.square(sobely)
    IXY = np.multiply(sobelx, sobely)

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    gaussian = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]) / 16

    X_filter = cv2.filter2D(IXX, cv2.CV_64F, gaussian)
    Y_filter = cv2.filter2D(IYY, cv2.CV_64F, gaussian)
    XY_filter = cv2.filter2D(IXY, cv2.CV_64F, gaussian)

    # step 4: compute determinant and trace of the M matrix
    det = X_filter * Y_filter - (XY_filter ** 2)
    trace = (X_filter + Y_filter)

    # step 5: compute R scores with k = 0.05
    k = 0.05
    R = det - k * (np.square(trace))

    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0

    # step 7: non-maximum suppression
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':
    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)

    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)

    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()

    # visualization for your debugging
    fig = plt.figure()

    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')

    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')

    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
