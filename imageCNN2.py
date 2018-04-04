#!/usr/bin/env python3
# course: TCSS555
# User profile in social media - imageCNN2
# date: 10/10/2017
# name: Team 4 - Iris Xia
# description: Python file to generate gender prediction from image using CNN by Levi and Hassner
import sys, getopt
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras import optimizers
from keras import applications
from keras import backend as K
import numpy as np
import pandas as pd
import relation_Iris

# ---------------------------------------------------------------------
# Method for gender prediction using CNN by Levi and Hassner
#
# @param inputfile: string, input file path
# @param width, height: int, input image size for the trained CNN model
# @return profile_test: pd.dataframe dataframe for gender prediction
# ---------------------------------------------------------------------
def predictGender(inputfile, width, height):
    # create dataframe for prediction
    test_profile_csv = pd.read_csv(inputfile + "profile/profile.csv", index_col=0)
    profile_test = pd.DataFrame(test_profile_csv,
                                columns=['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])

    img_width, img_height = width, height

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # create model, compile and load weights
    model = Sequential()
    model.add(Conv2D(96, (7, 7), strides=(4, 4), input_shape=input_shape, bias_initializer='zeros', padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))


    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    model.add(Conv2D(384, (3, 3), padding="same", bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))


    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.5, nesterov=True), metrics=['accuracy'])
    # end of load model
    model.load_weights("crop_model_best.hdf5")

    # try:
    #     os.remove(path1 + '/.DS_Store')
    # except OSError:
    #     pass

    # listing = os.listdir(path1)
    # num_samples = len(listing)
    # print('=======start predicting gender using CNN1\n=======')
    # print("  number of test images to be resized is: ", num_samples, "\n")
    
    #predict for each user in the test profile
    path1 = inputfile + "image"
    for index, row in profile_test.iterrows():
        userid = row['userid']
        img = load_img(path1 + '/' + userid + '.jpg', False, target_size=(img_width, img_height))
        x = img_to_array(img)/(255.0)
        #print(x)
        x = np.expand_dims(x, axis=0)
        print(userid)
        preds = model.predict_classes(x)
        probs = model.predict_proba(x)
        # userids = np.append(userids, filename[])
        # print(type(preds[0]))
        # print(type(probs))
        print(preds, probs, "\n")
        profile_test.ix[index, 'gender'] = preds[0]
        #profile_test.to_csv('genderhahaha.csv')

    return profile_test





