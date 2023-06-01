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
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)

    image_thres_mean = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    image_thres_gauss = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    image_thres_mean_blur = cv2.adaptiveThreshold(
        image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    image_thres_gauss_blur = cv2.adaptiveThreshold(
        image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    cv2.imshow("Source image", image_gray)
    cv2.imshow("Threshold Mean image", image_thres_mean)
    cv2.imshow("Threshold Gaussian image", image_thres_gauss)

    cv2.imshow("Blur Source image", image_blur)
    cv2.imshow("Blur Threshold Mean image", image_thres_mean_blur)
    cv2.imshow("Blur Threshold Gaussian image", image_thres_gauss_blur)

    cv2.waitKey(0)


def thresOtsu(image_color):
    image_gray = cv2.cvtColor(image_color[::2, ::2], cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)

    T = mahotas.thresholding.otsu(image_blur)  # Otsu

    temp = image_gray.copy()
    temp[temp > T] = 255
    temp[temp < 255] = 0
    temp = cv2.bitwise_not(temp)

    T = mahotas.thresholding.rc(image_blur)  # Riddler-Calvard
    temp2 = image_gray.copy()
    temp2[temp2 > T] = 255
    temp2[temp2 < 255] = 0
    temp2 = cv2.bitwise_not(temp2)

    result = np.vstack(
        [np.hstack([image_gray, image_blur]), np.hstack([temp, temp2])])

    cv2.imshow("Source image", image_gray)
    cv2.imshow("Threshold Mean image", result)

    cv2.waitKey(0)


def SodalEdgeDetection(sudoku):
    #sudoku = sudoku[::2,::2]

    sudoku_blur = cv2.GaussianBlur(sudoku, (3, 3), 1.0)

    grad_x = cv2.Sobel(sudoku, cv2.CV_64F, 1, 0, ksize=3,
                       borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(sudoku, cv2.CV_64F, 0, 1, ksize=3,
                       borderType=cv2.BORDER_DEFAULT)

    grad_x_blur = cv2.Sobel(sudoku_blur, cv2.CV_64F, 1,
                            0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    grad_y_blur = cv2.Sobel(sudoku_blur, cv2.CV_64F, 0,
                            1, ksize=3, borderType=cv2.BORDER_DEFAULT)

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
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1.0)

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
    resultado = np.vstack([np.hstack([img_gray, suave]),
                          np.hstack([canny1, canny2])])
    cv2.imshow("Detector de Bordas Canny", resultado)
    cv2.waitKey(0)

# Função para facilitar a escrita nas imagem


def escreve(img, texto, cor=(255, 0, 0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10, 20), fonte, 0.5, cor, 0, cv2.LINE_AA)

def cropMetadeImage(img):
    X_center = img.shape[1]//2
    Y_center = img.shape[0]//2
    Y_center_media = Y_center//2
    X_center_media = X_center//2

    img = img[Y_center-Y_center_media:Y_center+Y_center_media, X_center-X_center_media:X_center+X_center_media]
    return img

def identifyAndCountObjects(img1, img2, img3):
    # Passo 1: Conversão para tons de cinza
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # Passo 2: Blur/Suavização da imagem
    img_blur1 = cv2.blur(img_gray1, (5, 5))
    img_blur2 = cv2.blur(img_gray2, (5, 5))
    img_blur3 = cv2.blur(img_gray3, (5, 5))

    # Passo 3: Binarização resultando em pixels brancos e pretos
    thres1 = mahotas.thresholding.otsu(img_blur1)
    thres2 = mahotas.thresholding.otsu(img_blur2)
    thres3 = mahotas.thresholding.otsu(img_blur3)
    print(('-'*20)+"Thresholding"+('-'*20))
    print(thres1)
    print(thres2)
    print(thres3)
    bin1 = img_blur1.copy()
    bin2 = img_blur2.copy()
    bin3 = img_blur3.copy()
    # thres vai no lugar de 245, mas usei esse valor para achar somente os 6 dados da imagem e n seus valores também
    bin1[bin1 > 55] = 255
    bin1[bin1 < 255] = 0
    bin2[bin2 > 55] = 255
    bin2[bin2 < 255] = 0
    bin3[bin3 > 55] = 255
    bin3[bin3 < 255] = 0
    bin1 = cv2.bitwise_not(bin1)
    bin2 = cv2.bitwise_not(bin2)
    bin3 = cv2.bitwise_not(bin3)

    # Passo 4: Detecção de bordas com Canny
    edges1 = cv2.Canny(bin1, 70, 150)
    edges2 = cv2.Canny(bin2, 70, 150)
    edges3 = cv2.Canny(bin3, 70, 150)

    # Passo 5: Identificação e contagem dos contornos da imagem
    # cv2.RETR_EXTERNAL = conta apenas os contornos externos
    contours1, hierarchy1 = cv2.findContours(
        edges1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(
        edges2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv2.findContours(
        edges3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # A variável lx (lixo) recebe dados que não são utilizados

    # FINDCONTOURS -> É usada no OpenCV para encontrar os contornos em uma imagem.
    # image: A imagem binarizada de entrada, onde os contornos serão encontrados. Geralmente é uma imagem binária obtida após a aplicação de técnicas de limiarização ou detecção de bordas.
    # mode: O modo de recuperação dos contornos. Pode ser cv2.RETR_EXTERNAL para retornar apenas os contornos externos, cv2.RETR_LIST para retornar todos os contornos em uma lista simples, cv2.RETR_TREE para retornar todos os contornos em uma hierarquia completa, ou cv2.RETR_CCOMP para retornar todos os contornos em uma hierarquia de dois níveis.
    # method: O método de aproximação dos contornos. Pode ser cv2.CHAIN_APPROX_NONE para armazenar todos os pontos do contorno, cv2.CHAIN_APPROX_SIMPLE para armazenar apenas os pontos extremos do contorno, cv2.CHAIN_APPROX_TC89_L1 ou cv2.CHAIN_APPROX_TC89_KCOS para usar o algoritmo de aproximante de Freeman para reduzir o número de pontos do contorno.
    # contours: Uma lista vazia (ou None) que será preenchida com os contornos encontrados. Cada contorno é representado por um array NumPy.
    # hierarchy: Uma matriz (opcional) para armazenar a hierarquia dos contornos. É usado para representar a relação de aninhamento e ordem entre os contornos.
    # offset (opcional): Um deslocamento a ser adicionado a todos os pontos dos contornos.
    print('Countours 1')
    print(len(contours1))
    print(len(hierarchy1))
    print('Countours 2')
    print(len(contours2))
    print(len(hierarchy2))
    print('Countours 3')
    print(len(contours3))
    print(len(hierarchy3))

    escreve(img_gray1, "Imagem 1 em tons de cinza", 0)
    escreve(img_blur1, "Suavizacao da img 1 com Blur", 0)
    escreve(bin1, "Binarizacao da img 1 com Metodo Otsu", 255)
    escreve(edges1, "Detector de bordas Canny na img 1", 255)

    escreve(img_gray2, "Imagem 2 em tons de cinza", 0)
    escreve(img_blur2, "Suavizacao da img 2 com Blur", 0)
    escreve(bin2, "Binarizacao da img 2 com Metodo Otsu", 255)
    escreve(edges2, "Detector de bordas Canny na img 2", 255)

    escreve(img_gray3, "Imagem 3 em tons de cinza", 0)
    escreve(img_blur3, "Suavizacao da img 3 com Blur", 0)
    escreve(bin3, "Binarizacao da img 3 com Metodo Otsu", 255)
    escreve(edges3, "Detector de bordas Canny na img 3", 255)

    temp1 = np.vstack(
        [np.hstack([img_gray1, img_blur1]), np.hstack([bin1, edges1])])
    temp2 = np.vstack(
        [np.hstack([img_gray2, img_blur2]), np.hstack([bin2, edges2])])
    temp3 = np.vstack(
        [np.hstack([img_gray3, img_blur3]), np.hstack([bin3, edges3])])

    imgC2_1 = img1.copy()
    cv2.imshow("Imagem 1 Original", img1)
    cv2.drawContours(imgC2_1, contours1, -1, (255, 0, 0), 2)
    escreve(imgC2_1,"Valor total: "+ str(len(contours1))+" Pontos")
    imgC2_2 = img2.copy()
    cv2.imshow("Imagem 2 Original", img2)
    cv2.drawContours(imgC2_2, contours2, -1, (255, 0, 0), 2)
    escreve(imgC2_2,"Valor total: "+ str(len(contours2))+" Pontos")
    imgC2_3 = img3.copy()
    cv2.imshow("Imagem 3 Original", img3)
    cv2.drawContours(imgC2_3, contours3, -1, (255, 0, 0), 2)
    escreve(imgC2_3,"Valor total: "+ str(len(contours3))+" Pontos")

    # Baixando imagens 
    #cv2.imwrite('results/imagemTotalPontos1.jpg', imgC2_1)
    #cv2.imwrite('results/imagemTotalPontos2.jpg', imgC2_2)
    #cv2.imwrite('results/imagemTotalPontos3.jpg', imgC2_3)

    cv2.imshow(f"IMG 1 - Quantidade de objetos: {str(len(contours1))}", temp1)
    cv2.imshow("Resultado IMG 1", imgC2_1)
    cv2.imshow(f"IMG 2 - Quantidade de objetos: {str(len(contours2))}", temp2)
    cv2.imshow("Resultado IMG 2", imgC2_2)
    cv2.imshow(f"IMG 3 - Quantidade de objetos: {str(len(contours3))}", temp3)
    cv2.imshow("Resultado IMG 3", imgC2_3)
    cv2.waitKey(0)

    # DRAWCONTOURS -> É usada no OpenCV para desenhar contornos em uma imagem.
    # image: A imagem onde os contornos serão desenhados. Deve ser uma matriz NumPy (por exemplo, uma imagem carregada com cv2.imread()).
    # contours: A lista de contornos a serem desenhados. É uma lista de arrays NumPy, onde cada array representa um contorno, geralmente obtido pela função cv2.findContours().
    # contourIdx: Índice do contorno a ser desenhado. Se definido como -1, todos os contornos serão desenhados.
    # color: Cor dos contornos. Pode ser especificada como uma tupla (B, G, R) ou um valor escalar inteiro para imagens em tons de cinza.
    # thickness (opcional): Espessura da linha dos contornos. Se for um valor negativo ou cv2.FILLED, o contorno será preenchido.
    # lineType (opcional): Tipo de linha a ser usada para desenhar os contornos. Pode ser cv2.LINE_8, cv2.LINE_4, cv2.LINE_AA, etc.
    # hierarchy (opcional): A hierarquia dos contornos. É um array NumPy que representa a relação de hierarquia entre os contornos. Geralmente é obtido junto com os contornos pela função cv2.findContours().
    # maxLevel (opcional): O nível máximo da hierarquia a ser desenhado.
    # offset (opcional): Um deslocamento a ser aplicado a todos os pontos dos contornos.

def faceIdentificationIMG(img):
    #Converte para tons de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    #Criação do detector de faces
    #A OpenCV já possui o algoritmo pronto para detecção de Haar-like features, contudo, precisamos dos arquivo XML que é a fonte dos padrões para identificação dos objetos
    #df = cv2.CascadeClassifier('xml/haarcascade_frontalcatface.xml')
    df = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')
    #df = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')
    #df = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')

    #Executa a detecção
    faces = df.detectMultiScale(img_gray,
    scaleFactor = 1.05, minNeighbors = 7,
    minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
    humans =  df.detectMultiScale(img_gray, 1.9, 1)

    # A função detectMultiScale() é usada no OpenCV para realizar a detecção de objetos em uma imagem usando o algoritmo de Haar Cascade ou LBP (Local Binary Patterns). Essa função é comumente usada para detecção de faces, mas também pode ser aplicada a outras formas de detecção de objetos.

    # objects = cv2.detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
    # image: A imagem onde a detecção será realizada. Deve ser uma matriz NumPy (por exemplo, uma imagem carregada com cv2.imread()).
    # scaleFactor: O fator de escala usado para criar uma pirâmide de imagens em diferentes tamanhos. Esse fator especifica o quanto a imagem é reduzida a cada escala. Por exemplo, um valor de 1.1 significa uma redução de 10% em cada escala.
    # minNeighbors: O número mínimo de vizinhos que um objeto candidato deve ter para ser retido. Esse parâmetro afeta a qualidade e a precisão da detecção. Valores mais altos resultam em menos detecções falsas, mas também podem levar à perda de detecções verdadeiras.
    # flags: Parâmetros opcionais adicionais. Geralmente, é definido como cv2.CASCADE_SCALE_IMAGE.
    # minSize (opcional): O tamanho mínimo do objeto a ser detectado. Pode ser uma tupla (width, height) especificando a largura e altura mínimas.
    # maxSize (opcional): O tamanho máximo do objeto a ser detectado. Pode ser uma tupla (width, height) especificando a largura e altura máximas.

    #Desenha retangulos amarelos na iamgem original (colorida)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 7)

    # Salvando Imagem
    cv2.imwrite('results/imagemFaceDetectionGroup.jpg', img)

    #Exibe imagem. Título da janela exibe número de faces
    cv2.imshow(str(len(faces))+' face(s) encontrada(s).', img)
    cv2.waitKey(0)


def faceIdentificationWebCam():
    webCam = cv2.VideoCapture(0) # cv2.VideoCapture(0), onde 0 representa o índice da webcam
    #webCam = cv2.VideoCapture('assets/pedestrian.mp4')
    #webCam = cv2.VideoCapture('assets/industriWork.mp4')

    while True:
        #Converte para tons de cinza
        ret, frame = webCam.read()
        print(ret)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Criação do detector de faces
        #A OpenCV já possui o algoritmo pronto para detecção de Haar-like features, contudo, precisamos dos arquivo XML que é a fonte dos padrões para identificação dos objetos
        #df = cv2.CascadeClassifier('xml/haarcascade_frontalcatface.xml')
        #df = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')
        df = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')
        #df = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')
        

        #Executa a detecção
        faces = df.detectMultiScale(frame_gray,
        scaleFactor = 1.05, minNeighbors = 7,
        minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)

        # A função detectMultiScale() é usada no OpenCV para realizar a detecção de objetos em uma imagem usando o algoritmo de Haar Cascade ou LBP (Local Binary Patterns). Essa função é comumente usada para detecção de faces, mas também pode ser aplicada a outras formas de detecção de objetos.

        # objects = cv2.detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
        # image: A imagem onde a detecção será realizada. Deve ser uma matriz NumPy (por exemplo, uma imagem carregada com cv2.imread()).
        # scaleFactor: O fator de escala usado para criar uma pirâmide de imagens em diferentes tamanhos. Esse fator especifica o quanto a imagem é reduzida a cada escala. Por exemplo, um valor de 1.105 significa uma redução de 5% em cada escala.
        # minNeighbors: O número mínimo de vizinhos que um objeto candidato deve ter para ser retido. Esse parâmetro afeta a qualidade e a precisão da detecção. Valores mais altos resultam em menos detecções falsas, mas também podem levar à perda de detecções verdadeiras.
        # flags: Parâmetros opcionais adicionais. Geralmente, é definido como cv2.CASCADE_SCALE_IMAGE.
        # minSize (opcional): O tamanho mínimo do objeto a ser detectado. Pode ser uma tupla (width, height) especificando a largura e altura mínimas.
        # maxSize (opcional): O tamanho máximo do objeto a ser detectado. Pode ser uma tupla (width, height) especificando a largura e altura máximas.

        #Desenha retangulos amarelos na iamgem original (colorida)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 7)

        #Exibe imagem. Título da janela exibe número de faces
        escreve(frame, f'{len(faces)} Pessoas Encontradas', 0)
        cv2.imshow(f'WebCam', frame)
        if(len(faces) != 0 ):
            print(len(faces))
            cv2.imshow('Face Encontrada', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    image_color = cv2.imread('assets/onePiece.jpg')
    sudoku = cv2.imread('assets/sudoku.PNG', cv2.COLOR_BGR2GRAY)
    river = cv2.imread('assets/river.jpg')
    road = cv2.imread('assets/road.jpg')
    dices = cv2.imread('assets/dices.jpg')
    dados1 = cv2.imread('assets/dados1.jpg')
    dados2 = cv2.imread('assets/dados2.jpg')
    dados3 = cv2.imread('assets/dados3.jpg')
    dados6 = cv2.imread('assets/dados6.jpeg')
    jogoLetras = cv2.imread('assets/jogoLetras.jpg')
    checkers = cv2.imread('assets/checkers.jpg')
    face1 = cv2.imread('assets/face1.jpg')
    peoples = cv2.imread('assets/peaples.jpg')
    grupo_amigos_capao = cv2.imread('assets/grupo_amigos_capao.jpg')

    #dados2 = dados2[::2, ::2]
    img = peoples
    img1 = dados1
    img2 = cropMetadeImage(dados2)
    img3 = dados3
    

    #cv2.imshow('Source Image', img)
    #cv2.waitKey(0)

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

    # EdgeDetectionCanny(road)

    # identifyAndCountObjects(img1, img2, img3)

    # faceIdentificationIMG(img)

    faceIdentificationWebCam()


main()
