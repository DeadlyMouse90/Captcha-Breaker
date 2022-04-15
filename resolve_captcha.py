from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from image_treatment import image_treatment

def captcha_breaker():
    with open('labels_model.dat', 'rb') as file_translator:
        lb = pickle.load(file_translator)

    model = load_model('model_training.hdf5')

    image_treatment('resolve', 'resolve')

    files = list(paths.list_images('resolve'))
    for file in files:
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

        region_letters = sorted(region_letters, key=lambda x: x[0])

        image_final = cv2.merge([image] * 3)
        prevision = []

        for rectangle in region_letters:
            x, y, width, height = rectangle
            image_letter = image[y - 2:y + height + 2, x - 2:x + width + 2]

            image_letter = resize_to_fit(image_letter, 20, 20)

            image_letter = np.expand_dims(image_letter, axis=2)
            image_letter = np.expand_dims(image_letter, axis=0)

            letter_prediction = model.predict(image_letter)
            letter_prediction = lb.inverse_transform(letter_prediction)[0]
            prevision.append(letter_prediction)

        text_predict = ''.join(prevision)
        print(text_predict)

if __name__ == "__main__":
    captcha_breaker()