from image_utils import *
import os

def rename_dir_files(folder_path, initials, ext):
    list_imgs = get_img_list(folder_path, ext)
    for ii in range(len(list_imgs)):
        print('img_no = ', ii)
        img_path = list_imgs[ii];
        img_new_name = "%s%6.6d.%s" % (initials, ii, ext)
        img_new_path = os.path.join(folder_path, img_new_name)
        os.rename(img_path, img_new_path)

rename_dir_files("/media/user/DATA/Dataset/car_color/color/yellow", "img_", "jpg")
