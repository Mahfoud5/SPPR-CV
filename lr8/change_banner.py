# Реализация замены билболда на новый
import matplotlib
import numpy as np
from PIL import Image
import scipy.ndimage as sp
import cv2
from matplotlib import pyplot as plt
import os



def replace_pixels_with_image(img, im_out):
    mask = np.all(im_out == [0, 0, 0], axis=2)  # Создание маски для пикселей, которые нужно заменить
    im_out[mask] = img[mask]  # Замена пикселей
    return im_out


def change(img_path, banner_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    print('Укажите 4 точки билборда с левого верхнего угла и по часовой')
    coord = plt.ginput(4)
    plt.show()
    coordinates = [(int(x), int(y)) for x, y in coord]
    print('you clicked:', coordinates)
    street_points = np.array(coordinates)

    frame =cv2.cvtColor(cv2.imread(banner_path), cv2.COLOR_BGR2RGB)
    frame_points = np.array([[0, 0],[frame.shape[1], 0],[frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])

    # Рассчитайте гомографию
    h, status = cv2.findHomography(frame_points, street_points)

    # Трансформируем исходное изображение, используя полученную гомографию
    im_out = cv2.warpPerspective(frame.copy(), h, (img.shape[1],img.shape[0]))
    return replace_pixels_with_image(img, im_out)



if __name__ == '__main__' :
    os.chdir('Downloads//comp_vision//lr8')  # Изменение текущей директории на целевую
    img_path = 'image_1.jpg'
    banner_path = 'image_2.jpg'
    result = change(img_path, banner_path)
    plt.imshow(result)
    plt.show()