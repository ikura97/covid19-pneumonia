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


from sklearn import metrics


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
version_str = "xray1_kfold_all_tryout"
batch_size = 64
epochs = 20


folds_acc = []
folds_f1 = []
folds_roc = []
folds_pre = []

folds_conf = []
folds_sens = []
folds_spec = []
folds_jaccard = []
folds_mean_s = []
folds_log_loss = []


def listToString(s):


    str1 = ""


    for ele in s:
        str1 = str1 + ele + "\\"


    return str1


def predict_to_one_column(predict):
    """    
    Returns one column from predict results instead of 0 and 1 columns.
    """

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


data_c_p = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

data_c_p["category"] = data_c_p["category"].replace(
    {0: 'COVID19', 1: 'PNEUMONIA'})

y = data_c_p.drop(['filename'], axis=1)

X = data_c_p.drop(['category'], axis=1)


kf = KFold(n_splits=5, shuffle=True)
# j = 0

for train_idx, test_idx in kf.split(X, y):
    print("TRAIN:", train_idx, "TEST:", test_idx)
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    train_all = [x_train, y_train]
    train_df = pd.concat(train_all, axis=1, join='inner')



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


    nb_samples_of_train = x_train.shape[0]
    nb_samples_of_test = x_test.shape[0]

    print(
        f"number of samples of train in kfold number: {nb_samples_of_train} \n")
    print(
        f"number of samples of test in kfold number: {nb_samples_of_test} \n")

    # model starts

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
    # model.add(Dropout(0.2))
    # 178
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(
        0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('relu'))  # v26'da ayrıldı
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    optimizer = Adam(learning_rate=0.001)

    # serialize model to JSON
    model_json = model.to_json()
    with open('kfold_tryout_x1' + version_str + month_day_hour_extractor('Europe/Istanbul') + '{}.json', "w") as json_file:
        json_file.write(model_json)

    model.compile(loss="categorical_crossentropy", metrics=[
                  "accuracy"], optimizer=optimizer)
    model.summary()



    checkpoint_monitor = 'accuracy'
    earlystop_monitor = 'accuracy'
    lr_reduct_monitor = 'loss'
    earlystop_patience = 9  # 75
    lr_reduct_patience = 4  # 30
    i = 1

    """
    Callback setup for fitting the model. Returns callbacks according to 
    the parameters. ModelCheckPoint uses to current date from the function 
    of month_day_hour_extractor and saves it as an h5 file.
    """

    if (checkpoint_monitor == 'loss'):
        mode_mod_checkp = 'min'
    elif (checkpoint_monitor == 'accuracy'):
        mode_mod_checkp = 'max'

    if (earlystop_monitor == 'loss'):
        mode_mod_early = 'min'
    elif (earlystop_monitor == 'accuracy'):
        mode_mod_early = 'max'

    mc = ModelCheckpoint('labHB2_x1_model_from_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') +
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

    # callback ends

    # fitting phase starts

    history = model.fit(
        train_gen,
        epochs=epochs,

        steps_per_epoch=nb_samples_of_train//batch_size,  # total_train
        callbacks=callbacks
    )

    # fitting phase ends

    # Testing phase starts

    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        x_test,
        None,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )



    # Prediction Phase

    nb_samples = x_test.shape[0]

    try:
        predict = model.predict(test_generator,
                                steps=np.ceil(nb_samples/batch_size))

        one_col_predict = predict_to_one_column(predict)

        y_pred = one_col_predict

        # test_df['category'] = np.argmax(predict, axis=-1)

        label_map = dict((v, k)
                         for k, v in train_gen.class_indices.items())

        y_test['category'] = y_test['category'].replace(label_map)

        y_test_copy = y_test.copy()

        y_test_copy['category'] = y_test_copy['category'].replace({
            'PNEUMONIA': 1, 'COVID19': 0})

        # return one_col_predict, test_df

    except:
        print("An error occurred while predicting.")



    # yeni nesil metrics
    conf = metrics.confusion_matrix(y_test_copy, y_pred)
    accuracy = metrics.accuracy_score(y_test_copy, y_pred)
    pre = metrics.precision_score(y_test_copy, y_pred)


    sensitivity = metrics.recall_score(y_test_copy, y_pred)

    f1 = metrics.f1_score(y_test_copy, y_pred)
    roc_auc = metrics.roc_auc_score(y_test_copy, y_pred, multi_class='ovr')



    mean_squ_er = metrics.brier_score_loss(y_test_copy, y_pred)

    log_loss_s = metrics.log_loss(y_test_copy, y_pred)
    specificity = conf[1, 1]/(conf[1, 0]+conf[1, 1])



    folds_acc.append(accuracy)
    folds_f1.append(f1)
    folds_roc.append(roc_auc)
    folds_pre.append(pre)

    folds_conf.append(conf)
    folds_sens.append(sensitivity)
    folds_spec.append(specificity)

    folds_mean_s.append(mean_squ_er)
    folds_log_loss.append(log_loss_s)



    conf_df = pd.DataFrame(conf, columns=['0', '1'])

    only_one_fold_metrics = pd.DataFrame({
        'accuracy': round(accuracy, 5),
        'precision': round(pre, 5),
        'sensitivity': round(sensitivity, 5),
        'f1_score': round(f1, 5),
        'roc_auc_score': round(roc_auc, 5),
        'specificity': round(specificity, 5),
        'mean_squarred_error': round(mean_squ_er, 5),
        'log_loss': round(log_loss_s, 5)
    }, index=[i])

    print(only_one_fold_metrics)


general_metrics_including_all_folds = pd.DataFrame({
    'accuracy': folds_acc,
    'f1_score': folds_f1,
    'roc_auc_score': folds_roc,
    'precision': folds_pre,
    'sensitivity': folds_sens,
    'specificity': folds_spec,
    'mean_squarred_error': folds_mean_s,
    'log_loss': folds_log_loss,

})

print("X2 Final result of all k-folds: \n", general_metrics_including_all_folds)


#############################################################################
#---------------------------------------------------------------------------#
#############################################################################


def export_to_txt_csv_same_dir(df, desired_file_name): 


    try:
        path_str_txt = desired_file_name + "_" + \
            month_day_hour_extractor('Europe/Istanbul') + '.txt'
        with open(path_str_txt, 'a') as f:
            f.write(
                df.to_string(header=True, index=True)
            )


        path_str_csv = desired_file_name + '.csv'
        df.to_csv(path_str_csv, header=True, index=True, sep='\t', mode='a')
        print("Exporting to txt and csv file to same directory is successful.")
    except:
        return "An error occurred when exporting to files."


export_to_txt_csv_same_dir(
    general_metrics_including_all_folds, 'kfold_cikti_deneme')
