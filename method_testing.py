import cv2

methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV
]

image = cv2.imread('bdcaptcha/telanova250.png')

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

i = 0

for method in methods:
    i += 1
    _, image_treatment = cv2.threshold(image_gray, 127, 255, method or cv2.THRESH_OTSU)
    cv2.imwrite(f'test_method/image_treatment_{i}.png', image_treatment)

def getColor(img, x, y):
    return img.item(y, x, 0), img.item(y, x, 1), img.item(y, x, 2)

def setColor(img, x, y, r, g, b):
    img.itemset((y, x, 0), b)
    img.itemset((y, x, 1), g)
    img.itemset((y, x, 2), r)

    return img

obj_img = cv2.imread('test_method/image_treatment_3.png')
height, width, _ = obj_img.shape

for y in range(0, height):
    for x in range(0, width):
        b, g, r = getColor(obj_img, x, y)

        if b < 101 and g < 101 and r < 101:
            obj_img = setColor(obj_img, x, y, 0, 0, 0)
        else:
            obj_img = setColor(obj_img, x, y, 255, 255, 255)

cv2.imwrite('test_method/imagefinal.png', obj_img)