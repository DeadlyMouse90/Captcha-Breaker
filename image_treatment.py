import cv2
import os
import glob
from tqdm import tqdm

def getColor(img, x, y):
    return img.item(y, x, 0), img.item(y, x, 1), img.item(y, x, 2)


def setColor(img, x, y, r, g, b):
    img.itemset((y, x, 0), b)
    img.itemset((y, x, 1), g)
    img.itemset((y, x, 2), r)

    return img

def image_treatment(source_folder, destination_folder):
    files = glob.glob(f'{source_folder}/*')
    for file in tqdm(files):
        image = cv2.imread(file)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, img_treatment = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        file_name = os.path.basename(file)
        cv2.imwrite(f'{destination_folder}/{file_name}', img_treatment)

    files = glob.glob(f'{destination_folder}/*')
    for file in tqdm(files):
        image = cv2.imread(file)
        height, width, _ = image.shape

        for y in range(0, height):
            for x in range(0, width):
                b, g, r = getColor(image, x, y)

                if b < 115 and g < 115 and r < 115:
                    image = setColor(image, x, y, 0, 0, 0)
                else:
                    image = setColor(image, x, y, 255, 255, 255)
        file_name = os.path.basename(file)
        cv2.imwrite(f'{destination_folder}/{file_name}', image)

if __name__ == "__main__":
    image_treatment('bdcaptcha250', 'identified')
