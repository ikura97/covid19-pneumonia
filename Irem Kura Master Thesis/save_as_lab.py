from pathlib import Path  
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

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 = str1 + ele + "\\"
    
    # return string  
    return str1

 
try:
    filenames = glob("xray2" + "\P\*.*")
    filenames += glob("xray2" + "\C\*.*")

    categories = []
    for index, filename in enumerate(filenames):
        one_up = Path(filename).resolve().parents[0]
        if r'yeni_datasetler\xray2\P' in str(one_up):
            categories.append(1)
        elif r'yeni_datasetler\xray2\C' in str(one_up):
            categories.append(0)
except:
    print("An error occurred when getting train data.")

data_dict = {'filenames': filenames,
             'categories': categories}

data_df = pd.DataFrame(data_dict)
zero_count = data_df[data_df['categories'] == 0].count()
one_count = data_df[data_df['categories'] == 1].count()

lab = cv2.imread('xray2/C/01.jpEg')
lab_23 = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)   # BGR -> RGB


lab_images = []
lab_categories = []

for index, row in data_df.iterrows():
    temp_img = cv2.imread(row['filenames'])  
    img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
    
    img_name = re.split("\\\\", row['filenames'])[-1]    
    remaining_dir = re.split("\\\\", row['filenames'])[:-1]
    whole_dir = []
    whole_left_dir = listToString(remaining_dir)     
    
    if not os.path.exists("lab\\"+ whole_left_dir):
        os.makedirs("lab\\"+ whole_left_dir)
    cv2.imwrite("lab\\" + row['filenames'], img)

    lab = "lab\\"
    cv2.imwrite(lab + row['filenames'], img)
