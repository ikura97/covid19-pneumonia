import matplotlib.patches as mpatches
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from pathlib import Path  # Returns a Pathlib object
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from pathlib import Path  # Returns a Pathlib object
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
from tensorflow.keras import layers


cv2.__version__


tf.__version__

print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
version_str = "ct2_128_labHB_all_tryout" 
batch_size = 128
epochs = 35


def listToString(s):


    str1 = ""


    for ele in s:
        str1 = str1 + ele + "\\"

    # return string
    return str1


def predict_to_one_column(predict):


    final_predict_value = []
    for index, el in enumerate(predict):
        if el[0] > el[1]:
            final_predict_value.append(0)
        elif el[0] <= el[1]:  # elif el[0] < el[1]:
            final_predict_value.append(1)
    return final_predict_value


def month_day_hour_extractor(timezone):
    """
    Returns day, month(by text), and hour of the current day to use it later 
    for saving graphs, and, h5 file.
    Input : Timezone as string (exp: 'Europe/Istanbul')
    Output: String (exp: '_March_18_03_')
    """
    from datetime import datetime
    import pytz

    eu_ist_tz = pytz.timezone(timezone)
    datetime_ist = datetime.now(eu_ist_tz)
    ist_today_day_hour = datetime_ist.strftime("_%B_%d_%H_")

    return ist_today_day_hour


try:
    filenames = glob(r"labHB\yeni_datasetler\xray1" + "\PNEUMONIA\*.*")
    filenames += glob(r"labHB\yeni_datasetler\xray1" + "\COVID19\*.*")

    categories = []
    for index, filename in enumerate(filenames):
        one_up = Path(filename).resolve().parents[0]
        if r'labHB\yeni_datasetler\xray1\PNEUMONIA' in str(one_up):
            categories.append(1)
        elif r'labHB\yeni_datasetler\xray1\COVID19' in str(one_up):
            categories.append(0)
except:
    print("An error occurred when getting train data.")
   
    
    


data_dict = {'filenames': filenames,
             'categories': categories}

data_df = pd.DataFrame(data_dict)
zero_count = data_df[data_df['categories'] == 0]['filenames'].count()
one_count = data_df[data_df['categories'] == 1]['filenames'].count()

# --------------------------
#X = [filenames]
#Y = [categories]

data_c_p = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

data_c_p["category"] = data_c_p["category"].replace(
    {0: 'COVID19', 1: 'PNEUMONIA'})


test_size = 0.2
train_df, test_df = train_test_split(
    data_c_p, test_size=test_size, random_state=42)
train_df = train_df.reset_index(drop=True)
train_df, validate_df = train_test_split(
    data_c_p, test_size=test_size, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

nb_samples_of_train = train_df.shape[0]
nb_samples_of_test = test_df.shape[0]
nb_samples_of_val = validate_df.shape[0]


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1

)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    None,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)

val_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    None,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

nb_of_samples_test = test_df.shape[0]

# Test için Generator oluşturma

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    None,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

# model
model = Sequential()
model.add(Conv2D(32, (3, 3),  # 3), activation="LeakyReLU")
          input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # 64 den 32 oldu dropout eklend
# 169
model.add(Conv2D(64, (3, 3)))  # 3), activation="LeakyReLU")
model.add(layers.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))  # 3), activation="LeakyReLU")
model.add(layers.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

# v12'de eklendi   # v18 = 64->32
model.add(Conv2D(128, (3, 3)))  # , activation="LeakyReLU")
model.add(layers.LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))  # v12'de eklendi
# 178
model.add(Flatten())
model.add(Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation('relu'))  # v26'da ayrıldı
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('sigmoid'))

optimizer = Adam(learning_rate=0.001)

# serialize model to JSON
model_json = model.to_json()
with open('ct2_128labHB2_model_from_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') + '{}.json', "w") as json_file:
    json_file.write(model_json)

model.compile(loss="categorical_crossentropy", metrics=[
              "accuracy"], optimizer=optimizer)
model.summary()



checkpoint_monitor = 'val_accuracy'
earlystop_monitor = 'val_accuracy'
lr_reduct_monitor = 'val_loss'
earlystop_patience = 75
lr_reduct_patience = 30
i = 4



if (checkpoint_monitor == 'val_loss'):
    mode_mod_checkp = 'min'
elif (checkpoint_monitor == 'val_accuracy'):
    mode_mod_checkp = 'max'

if (earlystop_monitor == 'val_loss'):

    mode_mod_early = 'min'
elif (earlystop_monitor == 'val_accuracy'):
    mode_mod_early = 'max'


mc = ModelCheckpoint('ct2_128_labHB2_model_from_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') +
                     '{}.h5'.format(str(i+1)), monitor=checkpoint_monitor, mode=mode_mod_checkp, verbose=1, save_best_only=True)

# earlystop = EarlyStopping(patience=75, monitor = "val_accuracy") # patience = 75 patience = 100
earlystop = EarlyStopping(monitor=earlystop_monitor,
                          mode=mode_mod_early, verbose=1, patience=earlystop_patience)

# We will reduce the learning rate when then accuracy not increase for 30 steps
learning_rate_reduction = ReduceLROnPlateau(monitor=lr_reduct_monitor,
                                            patience=lr_reduct_patience,  # 30 50
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction, mc]




# Modeli fitliyoruz
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_samples_of_val//batch_size,  # total_validate
    steps_per_epoch=nb_samples_of_train//batch_size,  # total_train
    callbacks=callbacks
)



# Visualization starts

try:
    epoch_count = len(history.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epoch_count,
                             round(len(history.history['loss'])/8)))

    max_for_loss = 0
    if np.amax(history.history['loss']) > np.amax(history.history['val_loss']):
        max_for_loss = np.amax(history.history['loss'])
    else:
        max_for_loss = np.amax(history.history['val_loss'])
    ax1.set_yticks(np.arange(0, max_for_loss, 0.2))

    red_patch = mpatches.Patch(color='red', label='Validation loss - Min: ' +
                               str(round(np.amin(history.history['val_loss']), 4)))
    blue_patch = mpatches.Patch(color='blue', label='Training loss - Min: ' +
                                str(round(np.amin(history.history['loss']), 4)))
    black_patch = mpatches.Patch(color='black', label="Epoch count: "
                                 + str(epoch_count))

    ax1.legend(loc='best', shadow=True, handles=[
               red_patch, blue_patch, black_patch])

    # ---------------------------------------------------------------------
    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'],
             color='r', label="Validation accuracy")


    max_for_acc = 0
    if np.amax(history.history['accuracy']) > np.amax(history.history['val_accuracy']):
        max_for_acc = np.amax(history.history['accuracy'])
    else:
        max_for_acc = np.amax(history.history['val_accuracy'])


    ax2.set_xticks(np.arange(1, epoch_count,
                             round(len(history.history['accuracy'])/8)))
    ax2.set_yticks(np.arange(0, max_for_acc, 0.2))  # min_for_acc, 0.05

    red_patch_acc = mpatches.Patch(color='red', label='Validation accuracy - Max: '
                                   + str(round(np.amax(history.history['val_accuracy']), 4)))
    blue_patch_acc = mpatches.Patch(color='blue', label='Training accuracy - Max: '
                                    + str(round(np.amax(history.history['accuracy']), 4)))
    black_patch_acc = mpatches.Patch(color='black', label="Epoch count: "
                                     + str(epoch_count))

    ax2.legend(loc='best', shadow=True, handles=[red_patch_acc,
                                                 blue_patch_acc, black_patch_acc])


    ax1.figure.savefig('ct2-128labHB2_Acc & Loss_project_helper_' + version_str +
                       month_day_hour_extractor('Europe/Istanbul') + '{}.jpg'.format(str(i+1)))


    print("Check the graphs.")
except:
    print("An error occurred while visualizing.")

# Visualization ends


nb_samples = test_df.shape[0]

try:
    predict = model.predict(test_generator,
                            steps=np.ceil(nb_samples/batch_size))

    one_col_predict = predict_to_one_column(predict)


    label_map = dict((v, k)
                     for k, v in train_gen.class_indices.items())

    test_df['category'] = test_df['category'].replace(label_map)

    test_df['category'] = test_df['category'].replace({
        'PNEUMONIA': 1, 'COVID19': 0})


except:
    print("An error occurred while predicting.")

# Performance Metrics Starts


y_test = test_df.iloc[:, 1:].values
y_pred = one_col_predict

try:
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import log_loss

    conf = confusion_matrix(y_test, y_pred)
    total = sum(sum(conf))

    accuracy = (conf[0, 0]+conf[1, 1])/total
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    sensitivity = conf[0, 0]/(conf[0, 0]+conf[0, 1])
    specificity = conf[1, 1]/(conf[1, 0]+conf[1, 1])
    jaccard = jaccard_score(y_test, y_pred, average=None)
    mean_squ_er = mean_squared_error(y_test, y_pred)
    log_loss = log_loss(y_test, y_pred)

    jaccard_rounded = [round(el, 5) for el in jaccard]

    conf_df = pd.DataFrame(conf, columns=['0', '1'])

    metrics = pd.DataFrame({
        'accuracy': round(accuracy, 5),
        'precision': round(pre, 5),
        'recall': round(rec, 5),
        'f1_score': round(f1, 5),
        'roc_auc_score': round(roc_auc, 5),
        'sensivity': round(sensitivity, 5),
        'specificity': round(specificity, 5),
        'mean_squarred_error': round(mean_squ_er, 5),
        'log_loss': round(log_loss, 5)
    }, index=[i])



except:
    print("An error occurred while printing metrics.")

# Performance Metrics Ends

# Export phase starts


df = metrics
arr = jaccard
desired_file_name = version_str + '_metrics'




try:
    path_str_txt = desired_file_name + "_" + \
        month_day_hour_extractor('Europe/Istanbul') + '.txt'
    with open(path_str_txt, 'a') as f:
        f.write(
            df.to_string(header=True, index=True)
        )
        f.write('\n --------------------------------- \n')
        f.write("Confusion Matrix \n")
        f.write(
            conf_df.to_string(header=True, index=True)
        )
        f.write('\n --------------------------------- \n')
        f.write("Jaccard Score \n")
        f.write(' '.join(map(str, arr)))
    path_str_csv = desired_file_name + '.csv'
    df.to_csv(path_str_csv, header=True, index=True, sep='\t', mode='a')
    print("Exporting to txt and csv file to same directory is successful.")
except:
    print("An error occurred when exporting to files.")

# Export phase ends
