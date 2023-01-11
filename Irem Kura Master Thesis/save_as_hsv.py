# -*- coding: utf-8 -*-

from pathlib import Path  # Returns a Pathlib object
import pandas as pd
import tensorflow as tf
import cv2
from glob import glob
import re
import os

cv2.__version__

# from keras.preprocessing.image import ImageDataGenerator
tf.__version__


IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# XRAY1_DIR = ""
# XRAY2_DIR = ""
# CT1_DIR = ""

CT2_DIR = ""
X_DIR = ""


def listToString(s):

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 = str1 + ele + "\\"

    # return string"
    return str1


try:
    filenames = glob("xray1" + "\COVID19\*.*")
    filenames += glob("xray1" + "\PNEUMONIA\*.*")

    categories = []
    for index, filename in enumerate(filenames):
        one_up = Path(filename).resolve().parents[0]
        if r'xray1\PNEUMONIA' in str(one_up):
            categories.append(1)
        elif r'xray1\COVID19' in str(one_up):
            categories.append(0)
except:
    print("An error occurred when getting train data.")

data_dict = {'filenames': filenames,
             'categories': categories}

data_df = pd.DataFrame(data_dict)
zero_count = data_df[data_df['categories'] == 0].count()
one_count = data_df[data_df['categories'] == 1].count()

img23 = cv2.imread('xray1/COVID19/COVID19(0).jpg')
img23 = cv2.cvtColor(img23, cv2.COLOR_BGR2RGB)   # BGR -> RGB
image_23 = cv2.cvtColor(img23, cv2.COLOR_RGB2HSV)


hsv_images = []
hsv_categories = []



for index, row in data_df.iterrows():
    temp_img = cv2.imread(row['filenames'])
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2HSV)
    
    img_name = re.split("\\\\", row['filenames'])[-1]    
    remaining_dir = re.split("\\\\", row['filenames'])[:-1]
    whole_dir = []
    whole_left_dir = listToString(remaining_dir)     
    
    if not os.path.exists("hsv\\"+ whole_left_dir):
        os.makedirs("hsv\\"+ whole_left_dir)
    cv2.imwrite("hsv\\" + row['filenames'], img)

    hsv = "hsv\\"
    cv2.imwrite(hsv + row['filenames'], img)
