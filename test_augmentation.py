import cv2
from SimpleAugmentatations.GeometricalTransformations import Geometrical_Transformation

img_path = 'img_folder/iron_man.png'
img_aug_path = 'img_folder/iron_man_aug.png'
def test_flip():
    img = cv2.imread(img_path)
    obj_geo = Geometrical_Transformation(img)
    flip_img = obj_geo.flip_image('hv')
    cv2.imwrite(img_aug_path, flip_img)


test_flip()

