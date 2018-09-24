import cv2
import random
import numpy as np
from add_noise import add_gaussian_noise
from image_utils import get_width_height_img

def get_cropped_roi(list_instances):
    no_instances = len(list_instances)
    rand_roi_index = random.randint(0, no_instances-1)
    instance = list_instances[rand_roi_index]
    img_full_path, x_min, y_min, x_max, y_max = instance.img_full_path, instance.x_min, instance.y_min, instance.x_max, instance.y_max
    ref_img = cv2.imread(img_full_path)
    crop_roi = ref_img[y_min:y_max, x_min:x_max]
    cv2.resize(crop_roi, (0, 0), fx=1.2, fy=1.2)
    gauss_crop_roi = np.around(add_gaussian_noise(crop_roi))
    gauss_crop_roi = gauss_crop_roi.astype(np.int64)
    return gauss_crop_roi


def get_final_image(crop_roi, background_img, roi_2paste):
    rx_min, ry_min, rx_max, ry_max = roi_2paste[0], roi_2paste[1], roi_2paste[2], roi_2paste[3]
    x_min = random.randint(rx_min, rx_max)
    y_min = random.randint(ry_min, ry_max)

    crp_wd, crp_ht = get_width_height_img(crop_roi)
    bg_wd, bg_ht = get_width_height_img(background_img)
    if(x_min + crp_wd > bg_wd):
        x_min = (x_min + crp_wd - bg_wd - 1)

    x_max = x_min + crp_wd
    y_max = y_min + crp_ht
    final_image = background_img.copy()
    final_image[y_min:y_max, x_min:x_max] = crop_roi
    #cv2.rectangle(final_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    #cv2.imwrite('final_img.png', final_image)

    return final_image, [x_min, y_min, x_max, y_max]


