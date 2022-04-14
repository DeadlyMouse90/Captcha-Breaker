import cv2
import os
import glob
from tqdm import tqdm

files = glob.glob('identified/*')
for file in tqdm(files):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, new_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_letters = []

    for contour in contours:
        (x, y, width, height) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 115:
            region_letters.append((x, y, width, height))
    if len(region_letters) != 5:
        continue

    image_final = cv2.merge([image] * 3)

    i = 0
    for rectangle in region_letters:
        x, y, width, height = rectangle
        image_letter = image[y-2:y+height+2, x-2:x+width+2]
        i += 1
        file_name = os.path.basename(file).replace('.png', f'letter{i}.png')
        cv2.imwrite(f'letters/{file_name}', image_letter)
        cv2.rectangle(image_final, (x - 2, y - 2), (x + width + 2, y + height + 2), (0, 255, 0), 1)
    file_name = os.path.basename(file)
    cv2.imwrite(f'identified/{file_name}', image_final)