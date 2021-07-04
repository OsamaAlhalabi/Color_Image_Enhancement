import cv2 as cv
from image_enhancement import ImageEnhancement

if __name__ == '__main__':
    img = cv.imread('images/sample3.PNG')

    # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_enh = ImageEnhancement(gray_img, 2, 2, 1, False)
    # result = img_enh.enhance_gray_img()

    img_enh = ImageEnhancement(img, 2, 2, 1, True)
    result = img_enh.enhance_colored_img()
    cv.imshow('Original Image', img)
    cv.imshow('Enhanced Image', result)
    cv.waitKey(0)
