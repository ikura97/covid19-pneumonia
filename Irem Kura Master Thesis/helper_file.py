# -*- coding: utf-8 -*

import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator
tf.__version__
import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam

# NAMES CHAGE: json, h5, jpg, csv-txt

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

# train_df, train_generator, validate_df, validation_generator, nb_samples_of_train, nb_samples_of_val 

def generate_train_val_for_all(i, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size, test_size):
    
    """
        Takes train dataset and returns df and generated files of train and
        validation sets. 
        i is the dataset number given by the user.
        i = 1 for dataset of X-ray-1
        i = 2 for dataset of X-ray-2
        i = 3 for dataset of CT-1
        i = 4 for dataset of Tao
    """
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split
    from pathlib import Path # Returns a Pathlib object
    from glob import glob
    
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
       
    if i == 1:
        filenames = glob("train_x1" + "/*/*.*") 

        categories = []
        for index, filename in enumerate(filenames):
            path = Path(filename)
            if str(path.parent) == "train_x1\COVID19":
                categories.append(0)
            if str(path.parent) == "train_x1\PNEUMONIA":
                categories.append(1)
            
    elif i == 2:
        filenames = glob("train_x2" + "/*/*.*") 
    
        categories = []
        for index, filename in enumerate(filenames):
            path = Path(filename)
            if str(path.parent) == "train_x2\C":
                categories.append(0)
            if str(path.parent) == "train_x2\P":
                categories.append(1)
            
    elif i == 3:
        filenames = glob("train_ct1" + "/*/*.*")   
        categories = []
        for index, filename in enumerate(filenames):
            path = Path(filename)
            if str(path.parent) == "train_ct1\COVID2_CT":
                categories.append(0)
            if str(path.parent) == "train_ct1\pneumonia_CT":
                categories.append(1)  
           
    elif i == 4:
        try:            
            filenames = glob("train_tao" + "\CP5\*.*") 
            filenames += glob("train_tao" + "\COVID19\*\*.*") 
                    
            categories = []
            for index, filename in enumerate(filenames):
                one_up = Path(filename).resolve().parents[0]
                if r'train_tao\CP5' in str(one_up):
                    categories.append(1)
                elif r'train_tao\COVID19' in str(one_up):
                    categories.append(0)  
        except:
            print("An error occurred when getting train data.")         
    else:
        return "Please enter a valid train dataset - which is parameter i."
 
    data_c_p = pd.DataFrame({
            'filename': filenames,
            'category': categories
        })
        
    data_c_p["category"] = data_c_p["category"].replace({0: 'COVID19', 1: 'PNEUMONIA'}) 
    
    try:
        train_df, validate_df = train_test_split(data_c_p, test_size= test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)
    
        nb_samples_of_train = train_df.shape[0]
        nb_samples_of_val = validate_df.shape[0]
    
    except:
        return "An error occurred when train & validation splitting."
    
    try:
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
            batch_size= batch_size
        )
        
    except:
          return "An error occurred while generating train set."
    try:
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
        
    except:
        return "An error occurred while generating train set."      

    return train_df, train_gen, validate_df, val_generator, nb_samples_of_train, nb_samples_of_val 

def generate_test_for_all(i, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size):
        
    """
        Takes test set and returns df and generated files of test sets. 
        i is the dataset number given by the user.
        i = 1 for dataset of X-ray-1
        i = 2 for dataset of X-ray-2
        i = 3 for dataset of CT-1
        i = 4 for dataset of Tao
    """
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from pathlib import Path # Returns a Pathlib object
    from glob import glob
    
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
       
    if i == 1:
        try:
            filenames = glob("test_x1" + "/*/*.*") 

            categories = []
            for index, filename in enumerate(filenames):
                path = Path(filename)
                if str(path.parent) == "test_x1\COVID19":
                    categories.append(0)
                if str(path.parent) == "test_x1\PNEUMONIA":
                    categories.append(1)
        except:
            print("An error occurred while getting Xray-1 test data.")
    elif i == 2:
        try:
            filenames = glob("test_x2" + "/*/*.*") 
            categories = []
            for index, filename in enumerate(filenames):
                path = Path(filename)
                if str(path.parent) == "test_x2\C":
                    categories.append(0)
                if str(path.parent) == "test_x2\P":
                    categories.append(1)
        except:
            print("An error occurred while getting Xray-2 test data.")
          
    elif i == 3:
        try:
            filenames = glob("test_ct1" + "/*/*.*")     
            categories = []
            for index, filename in enumerate(filenames):
                path = Path(filename)
                if str(path.parent) == "test_ct1" + "\COVID2_CT":
                    categories.append(0)
                if str(path.parent) == "test_ct1" + "\pneumonia_CT":
                    categories.append(1)             
        except:
            print("An error occurred while getting Tao test data.")
   
    elif i == 4:       
        try:
            filenames = glob("test_tao" + "\CP5\*.*") 
            filenames += glob("test_tao" + "\COVID19\*\*.*")         
            categories = []
            for index, test_filename in enumerate(filenames):
                one_up = Path(test_filename).resolve().parents[0]
                if r'test_tao\CP5' in str(one_up):
                    categories.append(1)
                elif r'test_tao\COVID19' in str(one_up):
                    categories.append(0)    
        except:
            print("An error occurred while getting Tao test data.")
                   
    else:
        return "Please enter a valid test set - which is parameter i."
             
    test_df = pd.DataFrame({
        'filename' : filenames,
        'category' : categories,
    })
            
    nb_of_samples_test = test_df.shape[0]
    
    # Test için Generator oluşturma 
    
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        None, 
        x_col='filename',
        y_col=None,
        class_mode= None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )  

    return test_df, nb_of_samples_test, test_generator     
    

def windback_generate_model(version_str, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, initial_lr_for_adam):
    from tensorflow.keras.regularizers import l2
    
    """
    Generates model with (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS) featured
        images according to initial_lr_for_adam, returns model
    """
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = "relu", input_shape= (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))  # 64 den 32 oldu dropout eklend
    # 169
    model.add(Conv2D(64, (3, 3), activation = "relu")) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation = "relu")) # v12'de eklendi   # v18 = 64->32
    model.add(MaxPooling2D(pool_size=(2, 2))) # v12'de eklendi
    # 178
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('relu')) # v26'da ayrıldı
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
   
    optimizer = Adadelta(learning_rate = initial_lr_for_adam)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('model_from_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') + '{}.json', "w") as json_file:
        json_file.write(model_json)    
        
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    model.summary()   
    
    return model
 
def callback_creator(i, version_str, checkpoint_monitor, earlystop_monitor, lr_reduct_monitor,
                     earlystop_patience, lr_reduct_patience):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
th_day_hour_extractor and saves it as an h5 file.
       
 
    if (checkpoint_monitor == 'val_loss'):
        mode_mod_checkp = 'min'
    elif (checkpoint_monitor == 'val_accuracy'):
        mode_mod_checkp = 'max'
        
    if (earlystop_monitor == 'val_loss'):
    
        mode_mod_early = 'min'
    elif (earlystop_monitor == 'val_accuracy'):
        mode_mod_early = 'max'    
        
        
    mc = ModelCheckpoint('model_from_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') + '{}.h5'.format(str(i+1)), monitor=checkpoint_monitor, mode= mode_mod_checkp, verbose=1, save_best_only=True)
         
    # earlystop = EarlyStopping(patience=75, monitor = "val_accuracy") # patience = 75 patience = 100
    earlystop = EarlyStopping(monitor=earlystop_monitor, mode= mode_mod_early, verbose=1, patience= earlystop_patience)
    
    # We will reduce the learning rate when then accuracy not increase for 30 steps
    learning_rate_reduction = ReduceLROnPlateau(monitor= lr_reduct_monitor, 
                                                patience= lr_reduct_patience, # 30 50
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
                                 
    callbacks = [earlystop, learning_rate_reduction, mc]

    return callbacks

# Tao için batch_size = 128, epochs = 50
def fit_model(model, train_generator, nb_samples_of_train, validation_generator, nb_samples_of_val, batch_size, 
              callbacks, train_df, validate_df, not_fast_run, FAST_RUN: bool = False):
    


    try:
        # total_train = train_df.shape[0]
        # total_validate = validate_df.shape[0]             
        epochs= 75 if FAST_RUN else not_fast_run
        
        # Modeli fitliyoruz
        history = model.fit(
            train_generator, 
            epochs= epochs,
            validation_data=validation_generator,
            validation_steps=nb_samples_of_val//batch_size, # total_validate
            steps_per_epoch=nb_samples_of_train//batch_size, # total_train
            callbacks=callbacks
        )
        return history
            
    except:
        print("An error occured or interrupted.")
    
# Training için görselleştirme

def visualize_training_val_sets_loss_acc(history, i, version_str):
    

    
    
    import numpy as np
    import matplotlib.patches as mpatches
       
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
                                    str(round(np.amin(history.history['val_loss']),4)))
        blue_patch = mpatches.Patch(color='blue', label='Training loss - Min: ' +
                                    str(round(np.amin(history.history['loss']),4 )))
        black_patch = mpatches.Patch(color='black', label= "Epoch count: " 
                                     + str(epoch_count))
        
        ax1.legend(loc='best', shadow=True, handles=[red_patch, blue_patch,black_patch])
      
        # ---------------------------------------------------------------------
        ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
        ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
        
        
        max_for_acc = 0
        if np.amax(history.history['accuracy']) > np.amax(history.history['val_accuracy']):
            max_for_acc = np.amax(history.history['accuracy'])
        else:
            max_for_acc = np.amax(history.history['val_accuracy'])
        
      
        ax2.set_xticks(np.arange(1, epoch_count, 
                                 round(len(history.history['accuracy'])/8)))
        ax2.set_yticks(np.arange(0, max_for_acc, 0.2)) # min_for_acc, 0.05
        
        red_patch_acc = mpatches.Patch(color='red', label='Validation accuracy - Max: ' 
                                       + str(round(np.amax(history.history['val_accuracy']), 4)))
        blue_patch_acc = mpatches.Patch(color='blue', label='Training accuracy - Max: '
                                       + str(round(np.amax(history.history['accuracy']), 4)))
        black_patch_acc = mpatches.Patch(color='black', label= "Epoch count: " 
                                + str(epoch_count))
        
        ax2.legend(loc='best', shadow=True, handles=[red_patch_acc, 
                                        blue_patch_acc,black_patch_acc])
        
       
        ax1.figure.savefig('Acc & Loss_project_helper_' + version_str + month_day_hour_extractor('Europe/Istanbul') + '{}.jpg'.format(str(i+1)))
        
        
        print("Check the graphs.")
        return epoch_count 
    except:
        print("An error occurred while visualizing.")
        
def predict_to_one_column(predict):
    
    """    
    Returns one column from predict results instead of 0 and 1 columns.
    """
    
    final_predict_value = []
    for index, el in enumerate(predict):
        if el[0] > el[1]:
            final_predict_value.append(0)
        elif el[0] <= el[1]: # elif el[0] < el[1]:
            final_predict_value.append(1)
    return final_predict_value   
    
def prediction(model, train_generator, test_generator, test_df, batch_size): 
    
    """
        Returns prediction on trained model and re-labelled test_df category
    names.
    """    
    
    
    nb_samples = test_df.shape[0]
    
    try:           
        predict = model.predict(test_generator, 
                                    steps=np.ceil(nb_samples/batch_size))
                
        one_col_predict = predict_to_one_column(predict)       
        # test_df['category'] = np.argmax(predict, axis=-1)
        
        label_map = dict((v,k) 
                        for k,v in train_generator.class_indices.items())
        
        test_df['category'] = test_df['category'].replace(label_map)
        
        test_df['category'] = test_df['category'].replace({
                                        'PNEUMONIA': 1, 'COVID19': 0 })
    
   
        
        return one_col_predict, test_df
    
    except:
        print("An error occurred while predicting.")    
    
# Metrik kontrolleri

def give_me_metrics_report_of_prediction(y_test, y_pred, i): 
    
    """    
    Returns full report of metrics after prediction.    
    """
    
    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score, recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import jaccard_score        
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import log_loss  
        
        conf = confusion_matrix(y_test, y_pred)
        total=sum(sum(conf))
        accuracy=(conf[0,0]+conf[1,1])/total # from conf. matrix calculate accuracy
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr') 
        sensitivity = conf[0,0]/(conf[0,0]+conf[0,1])
        specificity = conf[1,1]/(conf[1,0]+conf[1,1])
        jaccard = jaccard_score(y_test, y_pred, average=None)
        mean_squ_er = mean_squared_error(y_test, y_pred)
        log_loss = log_loss(y_test, y_pred)       
                    
        jaccard_rounded = [round(el,5) for el in jaccard]
        
        conf_df = pd.DataFrame(conf, columns = ['0','1'])
            
        metrics = pd.DataFrame({                  
                    'accuracy': round(accuracy,5),
                    'precision': round(pre,5),
                    'recall': round(rec,5),
                    'f1_score': round(f1,5),
                    'roc_auc_score': round(roc_auc,5),
                    'sensivity': round(sensitivity,5),
                    'specificity': round(specificity,5),
                    'mean_squarred_error': round(mean_squ_er,5),
                    'log_loss': round(log_loss ,5)                   
                    },index=[i])
                          
        return conf_df, jaccard_rounded, metrics
    except:        
        return "An error occurred while printing metrics."

def export_to_txt_csv_same_dir(df, arr, conf_df, desired_file_name):

    """
        Exports selected dataframes into same file directory as 
        desired_file_name. First df generally is metrics and second df is
        confusion matrix of prediction evaluation.
    """    


    try:
        path_str_txt = desired_file_name + "_" + month_day_hour_extractor('Europe/Istanbul') + '.txt'
        with open(path_str_txt, 'a') as f:
            f.write(
                df.to_string(header = True, index = True)
            )       
            f.write('\n --------------------------------- \n')
            f.write("Confusion Matrix \n")
            f.write(
                conf_df.to_string(header = True, index = True)
            )    
            f.write('\n --------------------------------- \n') 
            f.write("Jaccard Score \n")
            f.write(' '.join(map(str, arr)))                        
        path_str_csv = desired_file_name + '.csv'
        df.to_csv(path_str_csv, header=True, index=True, sep='\t', mode='a')    
        print("Exporting to txt and csv file to same directory is successful.")
    except:
        return "An error occurred when exporting to files."
