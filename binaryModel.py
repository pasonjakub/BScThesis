# Imports for training the model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from keras.models import Model
from PIL import ImageFile

# Imports for saving the results
from keras.callbacks import History
import json
import os

""" Constants which are used in the training process of different models """

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path_224 = "D://Datasets//BenignMalicious//_Train"
valid_path_224 = "D://Datasets//BenignMalicious//_Valid"
test_path_224 = "D://Datasets//BenignMalicious//_Test"

train_path_299 = "D://Datasets//BenignMalicious_299//_Train"
valid_path_299 = "D://Datasets//BenignMalicious_299//_Valid"
test_path_299 = "D://Datasets//BenignMalicious_299//_Test"

class_labels = ['Benign', 'Malicious']

history_before = History()
history_after = History()

models = {
    "VGG_16": "binaryPrediciton_VGG16.csv",
    "ResNet50": "binaryPrediciton_ResNet50.csv",
    "Inception_V3": "binaryPrediciton_Inceptionv3.csv",
}

main_path = os.getcwd()


"""
    VGG16 binary model training
"""

trainBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path_224, target_size=(224, 224), classes=class_labels, batch_size=32)
validBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path_224, target_size=(224, 224), classes=class_labels, batch_size=32)
testBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path_224, target_size=(224, 224), classes=class_labels, batch_size=32, shuffle=False)

base_model = keras.applications.vgg16.VGG16()  # Loading the model with ImageNet weights from Keras
model = Sequential()
for layer in base_model.layers[:-1]:            # Add all the layers without the output layer
    model.add(layer)
for layer in model.layers:                      # Freeze all the layers i.e., don't allow weights to change
    layer.trainable = False
model.add(Dense(2, activation="sigmoid"))       # Add output layer with 2 outputs and sigmoid activation function


# Compiling the model after the changes to it were made
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with only the last layer unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=50, verbose=2, callbacks=history_before)

for layer in model.layers[-3:]:             # Unfreezing all fully-connected layers
    layer.trainable = True

# Compiling the model after the changes to it were made
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with only the fully-connected layers unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=50, verbose=2, callbacks=history_after)

# After the model was trained history, model and predictions of the model are saved for future use
history_dict = history_before.history
json.dump(history_dict, open("D://Binary//VGG16_before.json", 'w'))      # Save the first history in the form of a json file

history_dict = history_after.history
json.dump(history_dict, open("D://Binary//VGG16_after.json", 'w'))       # Save the second history in the form of a json file

os.chdir("D://Binary")
file = 'VGG16_binary.h5'
model.save(file)                                                            # Saving the trained model with adjusted weights

Y_predicted = model.predict(testBatch, verbose=1, steps=400)
y_predicted = np.argmax(Y_predicted, axis=1)
y_predicted = np.asarray([y_predicted])
np.savetxt('binaryPrediciton_VGG16.csv', y_predicted, delimiter=',')        # Saving the model predictions
os.chdir(main_path)

"""
    ResNet-50 binary model training
"""

trainBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input) \
    .flow_from_directory(directory=train_path_224, target_size=(224, 224), classes=class_labels, batch_size=32)
validBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input) \
    .flow_from_directory(directory=valid_path_224, target_size=(224, 224), classes=class_labels, batch_size=32)
testBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input) \
    .flow_from_directory(directory=test_path_224, target_size=(224, 224), classes=class_labels, batch_size=32, shuffle=False)

base_model = keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg', classes=2, weights='imagenet')
x = base_model.output                                           # Using ResNet-50 weights adjusted on ImageNet
x = Flatten()(x)                                                # Flattening to one dimension the output of the last ResNet block
x = Dense(512, activation='relu')(x)                            # Adding fully-connected layer after Flatten layer
predictions = Dense(2, activation='sigmoid')(x)                 # Add output layer with 2 outputs and sigmoid activation function
model = Model(inputs=base_model.input, outputs=predictions)     # The prepared model with final architecture

for layer in base_model.layers:     # Freeze all the layers i.e., don't allow weights to change
    layer.trainable = False

# Compiling the model after the changes to it were made
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with only the last layer unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=50, verbose=2, callbacks=history_before)

# https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
for layer in model.layers[:143]:        # Making sure that all the previous layers are frozen
    layer.trainable = False
for layer in model.layers[143:]:        # Unfreezing the last residual block with all the following layers
    layer.trainable = True

# Compiling the model after the changes to it were made
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with the last residual block and all the following layers unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=50, verbose=2, callbacks=history_after)

# After the model was trained history, model and predictions of the model are saved for future use
history_dict = history_before.history
json.dump(history_dict, open("D://Binary//ResNet50_before.json", 'w'))      # Save the first history in the form of a json file

history_dict = history_after.history
json.dump(history_dict, open("D://Binary//ResNet50_after.json", 'w'))       # Save the second history in the form of a json file

os.chdir("D://Binary")
file = 'ResNet50_binary.h5'
model.save(file)                                                            # Saving the trained model with adjusted weights

Y_predicted = model.predict(testBatch, verbose=1, steps=400)
y_predicted = np.argmax(Y_predicted, axis=1)
y_predicted = np.asarray([y_predicted])
np.savetxt('binaryPrediciton_ResNet50.csv', y_predicted, delimiter=',')        # Saving the model predictions
os.chdir(main_path)

"""
    InceptionV3 binary model training
"""

trainBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
    .flow_from_directory(directory=train_path_299, target_size=(299, 299), classes=class_labels, batch_size=32)
validBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
    .flow_from_directory(directory=valid_path_299, target_size=(299, 299), classes=class_labels, batch_size=32)
testBatch = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
    .flow_from_directory(directory=test_path_299, target_size=(299, 299), classes=class_labels, batch_size=32, shuffle=False)

base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet')
x = base_model.output                                           # Using InceptionV3 weights adjusted on ImageNet
x = Dense(1024, activation='relu')(x)                           # Adding fully-connected layer with ReLU activation function
predictions = Dense(2, activation='sigmoid')(x)                 # Add output layer with 9 outputs and softmax activation function
model = Model(inputs=base_model.input, outputs=predictions)     # The prepared model with final architecture

for layer in base_model.layers:     # Freeze all the layers i.e., don't allow weights to change
    layer.trainable = False

# Compiling the model after the changes to it were made
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with only the last layer unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=100, verbose=2, callbacks=history_before)


for layer in model.layers[:249]:        # Making sure that all the previous layers are frozen
    layer.trainable = False
for layer in model.layers[249:]:        # Unfreezing the two last inception blocks with all the following layers
    layer.trainable = True

# Compiling the model after the changes to it were made
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Training the model on the new data with the last residual block and all the following layers unfrozen
model.fit(trainBatch, steps_per_epoch=4, validation_data=validBatch, validation_steps=4, epochs=100, verbose=2, callbacks=history_after)

# After the model was trained history, model and predictions of the model are saved for future use
history_dict = history_before.history
json.dump(history_dict, open("D://Binary//InceptionV3_before.json", 'w'))      # Save the first history in the form of a json file

history_dict = history_after.history
json.dump(history_dict, open("D://Binary//InceptionV3_after.json", 'w'))       # Save the second history in the form of a json file

os.chdir("D://Binary")
file = 'InceptionV3_binary.h5'
model.save(file)                                                            # Saving the trained model with adjusted weights

Y_predicted = model.predict(testBatch, verbose=1, steps=400)
y_predicted = np.argmax(Y_predicted, axis=1)
y_predicted = np.asarray([y_predicted])
np.savetxt('binaryPrediciton_InceptionV3.csv', y_predicted, delimiter=',')        # Saving the model predictions
os.chdir(main_path)
