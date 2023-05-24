import cv2
import mahotas
import numpy as np
from matplotlib import pyplot as plt
from copy import copy


def manipulandoPixelColor(image_color):
    image_color_1 = copy(image_color)
    image_color_2 = copy(image_color)
    image_color_3 = copy(image_color)
    for y in range(0, image_color_1.shape[0]):
        for x in range(0, image_color_1.shape[1]):
            image_color_1[y, x] = (255, 0, 0)  # BGR

    for y in range(0, image_color_2.shape[0], 10):
        for x in range(0, image_color_2.shape[1], 10):
            image_color_2[y:y+1, x:x+1] = (0, 255, 255)  # BGR

    image_color_3[300:400, 50:150] = (255, 255, 0)  # BGR

    #cv2.imshow('Image Colored', image_color_1)
    #cv2.imshow('Image drawed', image_color_2)
    cv2.imshow('Image drawed', image_color_3)
    cv2.waitKey()

def drawGeometricFormsText(image_color):
    vermelho = (0, 0, 255)
    verde = (0, 255, 0)
    azul = (255, 0, 0)

    image_color_1 = copy(image_color)

    # Draw a line on image
    cv2.line(image_color_1, (0, 0), (100, 200), verde, 10)
    # Draw a rectangle on image
    cv2.rectangle(image_color_1, (300, 300), (600, 600), azul, 3)
    # Draw a circle
    for raio in range(0, image_color_1.shape[0]//2, 20):
        cv2.circle(
            image_color_1, (image_color_1.shape[1]//2, image_color_1.shape[0]//2), raio, vermelho, 1)

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image_color_1, "TEXTO MANEIRO", (40, 80),
                font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_color_1, "TEXTO IRADO", (40, 150),
                font, 2, (0, 0, 0), 2, cv2.LINE_8)

    cv2.imshow('Drawed image', image_color_1)
    cv2.waitKey()

def shadowFilterImage(image_color):
    image_color_1 = copy(image_color)
    for y in range(0, image_color_1.shape[0]):
        for x in range(0, image_color_1.shape[1]):
            image_color_1[y, x] = image_color_1[y, x] // 5
    cv2.imshow("Blur", image_color_1)
    cv2.waitKey()

def cropResizeImage(image_color):
    image_color_1 = copy(image_color)

    X_center = image_color.shape[1]//2
    Y_center = image_color.shape[0]//2

    crop = image_color_1[Y_center-200:Y_center +
                         200, X_center-200:X_center+200]  # (Y, X)

    cv2.imshow("SRC image", image_color_1)
    cv2.imshow("Cropped image", crop)
    cv2.waitKey()

def flipImage(image_color):
    image_color_flip = copy(image_color)
    image_color_flip = cv2.flip(image_color, -1)
    cv2.imshow("Flipped image", image_color_flip)
    cv2.imshow("Source image", image_color)
    cv2.waitKey(0)

def rotateImage(image_color):
    img_height = image_color.shape[0]  # get height and width
    img_width = image_color.shape[1]
    center = (img_width//2, img_height//2)  # Find center

    # Rotate from center in 30° and scale 1.0
    rotated_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    image_color_rotate = cv2.warpAffine(
        image_color, rotated_matrix, (img_width, img_height))

    cv2.imshow("Source Image", image_color)
    cv2.imshow("Rotated Image", image_color_rotate)

    cv2.waitKey(0)

def makeMask(image_color):
    mask = np.zeros(image_color.shape[:2], dtype="uint8")

    (cX, cY) = (image_color.shape[1]//2, image_color.shape[0]//2)

    cv2.circle(mask, (cX, cY), 300, 255, -1)

    image_color_mask = cv2.bitwise_and(image_color, image_color, mask=mask)

    cv2.imshow("Mask Image", image_color_mask)
    cv2.waitKey(0)

def colorScales(image_color):
    image_color = image_color[::2, ::2]
    cv2.imshow("Source Image", image_color)

    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)

    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", hsv)

    lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2LAB)
    cv2.imshow("LAB Image", lab)

    cv2.waitKey(0)

def splitGrayColorChannels(image_color):
    image_color = image_color[::2, ::2]  # reduce image size
    (blue_channel, green_channel, red_channel) = cv2.split(image_color)

    cv2.imshow("Source Channel", image_color)
    cv2.imshow("Red Channel", red_channel)
    cv2.imshow("Gree Channel", green_channel)
    cv2.imshow("Bluee Channel", blue_channel)
    cv2.waitKey(0)

def splitColorChannels(image_color):
    image_color = image_color[::2, ::2]  # reduce image size
    (blue_channel, green_channel, red_channel) = cv2.split(image_color)

    zeros = np.zeros(image_color.shape[:2], dtype='uint8')

    cv2.imshow("Red image", red_channel)
    cv2.imshow("Red channel image", cv2.merge([zeros, zeros, red_channel]))

    cv2.imshow("Green image", green_channel)
    cv2.imshow("Green channel image", cv2.merge([zeros, green_channel, zeros]))

    cv2.imshow("Blue image", blue_channel)
    cv2.imshow("Blue channel image", cv2.merge([blue_channel, zeros, zeros]))

    cv2.waitKey(0)

def histogramEqualization(image_color):
    image_color = image_color[::2, ::2]
    image_gray = cv2.cvtColor(
        image_color, cv2.COLOR_BGR2GRAY)  # convert to gray

    def grayHist(image_gray):
        cv2.imshow("Gray image", image_gray)
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Gray image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(0)

    def colorHist(image_color):
        cv2.imshow("Colored image", image_color)

        channels = cv2.split(image_color)
        colors = ("b", "g", "r")

        plt.figure()
        plt.title("Color image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")

        for (channel, color) in zip(channels, colors):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.show()
        cv2.waitKey(0)

    def equalizationHist(image_gray):
        hist_eq = cv2.equalizeHist(image_gray)

        plt.figure()
        plt.title('Equalized Histogram')
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")
        plt.hist(hist_eq.ravel(), 256, [0, 256])
        plt.xlim([0, 256])
        plt.show()

        plt.figure()
        plt.title('Color Histogram')
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")
        plt.hist(image_gray.ravel(), 256, [0, 256])
        plt.xlim([0, 256])
        plt.show()

        cv2.waitKey()

    grayHist(image_gray)
    colorHist(image_color)
    equalizationHist(image_gray)

def smoothingImage(image_color):
    image_color = image_color[::3, ::3]

    blur = np.vstack([
        np.hstack([image_color, cv2.blur(image_color, (3, 3))]),
        np.hstack([cv2.blur(image_color, (5, 5)),
                  cv2.blur(image_color, (7, 7))]),
        np.hstack([cv2.blur(image_color, (9, 9)),
                  cv2.blur(image_color, (11, 11))]),
    ])

    gaussian_blur = np.vstack([
        np.hstack([image_color, cv2.GaussianBlur(image_color, (3, 3), 0)]),
        np.hstack([cv2.GaussianBlur(image_color, (5, 5), 0),
                  cv2.GaussianBlur(image_color, (7, 7), 0)]),
        np.hstack([cv2.GaussianBlur(image_color, (9, 9), 0),
                  cv2.GaussianBlur(image_color, (11, 11), 0)]),
    ])

    media_blur = np.vstack([
        np.hstack([image_color, cv2.medianBlur(image_color, 3)]),
        np.hstack([cv2.medianBlur(image_color, 5),
                  cv2.medianBlur(image_color, 7)]),
        np.hstack([cv2.medianBlur(image_color, 9),
                  cv2.medianBlur(image_color, 1)]),
    ])

    bilateral_blur = np.vstack([
        np.hstack([image_color, cv2.bilateralFilter(image_color, 3, 21, 21)]),
        np.hstack([cv2.bilateralFilter(image_color, 5, 35, 35),
                  cv2.bilateralFilter(image_color, 7, 49, 49)]),
        np.hstack([cv2.bilateralFilter(image_color, 9, 63, 63),
                  cv2.bilateralFilter(image_color, 11, 77, 77)]),
    ])

    cv2.imshow("Blur image", blur)
    cv2.imshow("Gaussian Blur image", gaussian_blur)
    cv2.imshow("Media Blur image", media_blur)
    cv2.imshow("Bilateral Blur image", bilateral_blur)
    cv2.waitKey(0)

def binarizationLimiar(image_color):
    image_gray = cv2.cvtColor(image_color[::2, ::2], cv2.COLOR_BGR2GRAY)

    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)

    (T, bin) = cv2.threshold(image_blur, 160, 255, cv2.THRESH_BINARY)
    (T, bin2) = cv2.threshold(image_blur, 160, 255, cv2.THRESH_BINARY_INV)

    """ result = np.vstack([
        np.hstack([image_blur, bin]),
        np.hstack([bin2, cv2.bitwise_and(image_gray, image_gray, mask=bin2)])
    ]) """

    cv2.imshow("Bin", bin)
    cv2.imshow("Bin2", bin2)
    cv2.waitKey(0)

def adaptativeThres(image_color):
    image_gray = cv2.cvtColor(image_color[::2, ::2], cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7,7), 0)

    image_thres_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    image_thres_gauss = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    image_thres_mean_blur = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    image_thres_gauss_blur = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    cv2.imshow("Source image", image_gray);
    cv2.imshow("Threshold Mean image", image_thres_mean);
    cv2.imshow("Threshold Gaussian image", image_thres_gauss);

    cv2.imshow("Blur Source image", image_blur);
    cv2.imshow("Blur Threshold Mean image", image_thres_mean_blur);
    cv2.imshow("Blur Threshold Gaussian image", image_thres_gauss_blur);

    cv2.waitKey(0)

def thresOtsu(image_color):
    image_gray = cv2.cvtColor(image_color[::2, ::2], cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7,7), 0)

    T = mahotas.thresholding.otsu(image_blur) # Otsu

    temp = image_gray.copy()
    temp[temp>T]=255
    temp[temp<255]=0
    temp = cv2.bitwise_not(temp)

    T=mahotas.thresholding.rc(image_blur) # Riddler-Calvard
    temp2 = image_gray.copy()
    temp2[temp2>T]=255
    temp2[temp2<255]=0
    temp2 = cv2.bitwise_not(temp2)

    result = np.vstack([np.hstack([image_gray, image_blur]), np.hstack([temp, temp2])])

    cv2.imshow("Source image", image_gray)
    cv2.imshow("Threshold Mean image", result)

    cv2.waitKey(0)

def SodalEdgeDetection(sudoku):
    #sudoku = sudoku[::2,::2]

    sudoku_blur = cv2.GaussianBlur(sudoku, (3,3), 1.0)

    grad_x = cv2.Sobel(sudoku, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(sudoku, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)

    grad_x_blur = cv2.Sobel(sudoku_blur, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y_blur = cv2.Sobel(sudoku_blur, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)

    grad_x = np.uint8(np.absolute(grad_x))
    grad_y = np.uint8(np.absolute(grad_y))

    grad_x_blur = np.uint8(np.absolute(grad_x_blur))
    grad_y_blur = np.uint8(np.absolute(grad_y_blur))

    image_sobel = cv2.bitwise_or(grad_x, grad_y)

    image_sobel_blur = cv2.bitwise_or(grad_x_blur, grad_y_blur)

    result = np.vstack([np.hstack([sudoku, grad_x, grad_y, image_sobel]),
                        np.hstack([sudoku_blur, grad_x_blur, grad_y_blur, image_sobel_blur])])
    
    cv2.imshow("Result", result)
    cv2.waitKey(0)

def filtroLapraceEdgeDetection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 1.0)

    lap = cv2.Laplacian(img_blur, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    res = np.hstack([img_gray, lap])
    cv2.imshow('Filtro Laplaciano', res)
    cv2.waitKey(0)

def EdgeDetectionCanny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(img_gray, (7, 7), 0)
    canny1 = cv2.Canny(suave, 20, 120)
    canny2 = cv2.Canny(suave, 70, 200)
    resultado = np.vstack([np.hstack([img_gray, suave ]),np.hstack([canny1, canny2])]) 
    cv2.imshow("Detector de Bordas Canny", resultado)
    cv2.waitKey(0)


def main():
    image_color = cv2.imread('assets/onePiece.jpg')
    sudoku = cv2.imread('assets/sudoku.PNG', cv2.COLOR_BGR2GRAY)
    river = cv2.imread('assets/river.jpg')
    road = cv2.imread('assets/road.jpg') 
    # cv2.imshow('Source Image', image_color)

    print(image_color.shape)
    print('--'*30)
    print(image_color.shape[0])
    print('--'*30)
    print(image_color.shape[1])

    # manipulandoPixelColor(image_color)

    # drawGeometricFormsText(image_color)

    # shadowFilterImage(image_color)

    # cropResizeImage(image_color)

    # flipImage(image_color)

    # rotateImage(image_color)

    # makeMask(image_color)

    # colorScales(image_color)

    # splitGrayColorChannels(image_color)

    # splitColorChannels(image_color)

    # histogramEqualization(image_color)

    # smoothingImage(image_color)

    # binarizationLimiar(image_color)

    # adaptativeThres(image_color)

    # thresOtsu(image_color)

    # SodalEdgeDetection(sudoku)

    # filtroLapraceEdgeDetection(road)

    EdgeDetectionCanny(road)

main()
