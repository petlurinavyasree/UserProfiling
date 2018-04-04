#!/usr/bin/env python3
# course: TCSS555
# User profile in social media - imageCNN3
# date: 10/10/2017
# name: Team 4 - Iris Xia
# description: Python file to generate gender prediction from image using CNN with VGG16 bottleneck features
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
# Method for gender prediction using CNN with VGG16 bottleneck features
#
# @param inputfile: string, input file path
# @param width, height: int, input image size for the trained CNN model
# @return profile_test: pd.dataframe dataframe for gender prediction
# ---------------------------------------------------------------------
def predictGender(inputfile, width, height):
    # result set
    test_profile_csv = pd.read_csv(inputfile + "profile/profile.csv", index_col=0)
    profile_test = pd.DataFrame(test_profile_csv,
                                columns=['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])

    img_width, img_height = width, height

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # create model, compile and load weights
    input_tensor = Input(shape=(224, 224, 3))
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    top_model.load_weights("weights-improvement-gender-94-0.81.hdf5")

    model = Model(input=base_model.input, output=top_model(base_model.output))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    # end of load model

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
        x = img_to_array(img)
        #print(x)
        x = np.expand_dims(x, axis=0)
        print(userid)
        preds = model.predict(x)
        # userids = np.append(userids, filename[])
        # print(type(preds[0]))
        # print(type(probs))
        print(preds, "\n")
        profile_test.ix[index, 'gender'] = preds
        #profile_test.to_csv('genderhahaha.csv')

    return profile_test




