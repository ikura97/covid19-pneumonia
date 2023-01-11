# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# tf.__version__

#import project_helper_module as phf
import helper_file as phf # deneme0

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
batch_size = 64
epochs = 20
test_size = 0.2
i= 1

dataset_number = 1

if dataset_number == 1: # Xray-1
    version_for_experiment = "xray1_all_tryout"
elif dataset_number == 2: # Xray-2
    version_for_experiment = "xray2_all_tryout"
elif dataset_number == 3: # CT-1
    version_for_experiment = "ct1_all_tryout"
elif dataset_number == 4: # Tao
    version_for_experiment = "tao_all_tryout"
else:
    print("Please enter a valid i (dataset code) number")

# Getting data, creating required train and val df and generators.
train_df, train_generator, validate_df, validation_generator, nb_samples_of_train, nb_samples_of_val = phf.generate_train_val_for_all(dataset_number, IMAGE_WIDTH, IMAGE_HEIGHT,batch_size,0.2) ##

# Creating model which will be used for training.
model = phf.windback_generate_model(version_for_experiment, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, 0.001)

# Creating callbacks will be used in train phase, according to the arguments.
callbacks = phf.callback_creator(i, version_for_experiment, 'val_accuracy', 'val_accuracy', 'val_loss', 75, 30)

# Fitting of the model which is "training"
history = phf.fit_model(model, train_generator, nb_samples_of_train, validation_generator, nb_samples_of_val, batch_size, callbacks, train_df, validate_df, epochs)

# Visualization of training results
epoch_count = phf.visualize_training_val_sets_loss_acc(history, i, version_for_experiment)

# Creating test df and generator for testing.
test_df, nb_of_samples_test, test_generator = phf.generate_test_for_all(dataset_number, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size)

# Prediction phase with test data. Traing_generator used for labels which
# are 0: Covid, 1: Pneumonia 
one_col_predict, test_df = phf.prediction(model, train_generator, test_generator, test_df, batch_size)

# Metrics report for prediction.
conf_df, jaccard, metrics = phf.give_me_metrics_report_of_prediction(test_df.iloc[:,1:].values, one_col_predict, i)

# Export prediction results into both csv and text file.
phf.export_to_txt_csv_same_dir(metrics, jaccard, conf_df, version_for_experiment + '_metrics')
