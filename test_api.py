import csv
import os
import sys
import numpy as np
import math

##my packages
sys.path.append('./img_utils')
from csv_to_yolo import *

in_dir = 'path_image_folder'
out_dir = 'final_out_path'
gt_csv_file_path = 'path_ground_truth/lbl_data'

convert_csv_to_yolo_format(in_dir, out_dir, gt_csv_file_path)