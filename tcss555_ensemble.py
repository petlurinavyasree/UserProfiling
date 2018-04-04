#!/usr/bin/env python
# course: TCSS555
# User profile in social media - ensemble
# date: 10/10/2017
# name: Team 4 - Iris Xia
# description: Python file to generate ensemble prediction from three sources
import pandas as pd
import relation_Iris
from RelationPredictor_Final import RelationPredictor
from prediction_manish import predict_gender
from imageCNN1 import predictGender
#from imageCNN2 import predictGender
#from imageCNN3 import predictGender
finaloutput = ['', '', '', '', '', '', ''] #store output for each user
output_frame = pd.DataFrame() #output dataframe


# ------------------------------------------------------------------
# Method for generating the base line result, by averaging the value
# for each feature using 9500 public training data.
# ------------------------------------------------------------------
def get_base_line():
    profile_csv = pd.read_csv("/data/training/profile/profile.csv", index_col=0)
    profile_frame = pd.DataFrame(profile_csv, columns=['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])

    age = [0, 0, 0, 0]
    age[0] = len(profile_frame[profile_frame['age'] < 25].index)
    age[1] = len(profile_frame[(profile_frame['age'] > 24) & (profile_frame['age'] < 35)].index)
    age[2] = len(profile_frame[(profile_frame['age'] > 34) & (profile_frame['age'] < 50)].index)
    age[3] = len(profile_frame[profile_frame['age'] > 49].index)
    print(age[0], ' ', age[1], ' ', age[2], ' ', age[3])
    group = age.index(max(age))
    if group == 0:
        age = 24
    elif group == 1:
        age = 34
    elif group == 2:
        age = 49
    else:
        age = 50

    gender = profile_frame['gender'].mean()
    ope = profile_frame['ope'].mean()
    con = profile_frame['con'].mean()
    ext = profile_frame['ext'].mean()
    agr = profile_frame['agr'].mean()
    neu = profile_frame['neu'].mean()
    baseline = [age, gender, ope, con, ext, agr, neu]
    return baseline

# -------------------------------
# Helper method for age grouping.
#
# @param age: int, user age
# -------------------------------
def age_helper(age):
    if age < 25:
        finaloutput[0] = 'XX-24'
    elif age < 35:
        finaloutput[0] = '25-34'
    elif age < 50:
        finaloutput[0] = '35-49'
    else:
        finaloutput[0] = '50-XX'

# -------------------------------------
# Helper method for generate age label.
#
# @param gender: int, user gender
# -------------------------------------
def gender_helper(gender):
    majgender = round(gender)
    if majgender == 0:
        finaloutput[1] = 'male'
    else:
        finaloutput[1] = 'female'

# -------------------------------------
# Print method for baseline results
# -------------------------------------
def print_baseline():
    baseline = get_base_line()

    print('\n\n\n\n\n====================================')
    print('|          AMONG', 9500, 'USERS        |')
    print('====================================')
    print(' [  Majority age  ] is ', baseline[0])
    print(' [ Majority gender] is ', baseline[1])
    print(' [  Avg ope score ] is', baseline[2])
    print(' [  Avg con score ] is', baseline[3])
    print(' [  Avg ext score ] is', baseline[4])
    print(' [  Avg agr score ] is', baseline[5])
    print(' [  Avg neu score ] is', baseline[6])
    print('------------------------------------\n\n\n\n\n')

# -------------------------------------------------------------
# Main ensemble method that combine results from three sources.
# Reading testing data from input files, generating the output
# .xml files including prediction results for each user in a
# proposed format into the output file.
#
# @param inputfile: string, input file path
# @param outputfile: string, output file path
# -------------------------------------------------------------
def ensemble(inputfile, outputfile):
    #generate base line
    baseline = get_base_line()
    print_baseline()

    #generate relation_Iris
    print('====================================\n')
    print('generate result from relation-Iris\n')
    print('====================================\n')
    relation_iris_frame = relation_Iris.get_result_relation(inputfile)
    print('================end==================\n')

    #generate output frame
    global output_frame
    output_frame = relation_iris_frame

    #generate relation_Navya
    print('============================================\n')
    print('generate result from RelationPredictor_Final\n')
    print('============================================\n')
    #relation_gender_frame = relation_gender.predictGender(inputfile)
    rp = RelationPredictor()
    predictedValues = rp.relationPrediction(inputfile)
    print('====================end=====================\n')

    #generate imageCNN
    print('====================================\n')
    print('generate result from CNN_gender\n')
    print('====================================\n')
    image_CNN1_frame = predictGender(inputfile, 150, 150)
    #image_CNN2_frame = predictGender(inputfile, 227, 227)
    #image_CNN3_frame = predictGender(inputfile, 224, 224)
    print('================end=================\n')

    #generate prediction_manish
    print('====================================\n')
    print('generate result from prediction_manish\n')
    print('====================================\n')
    predictedValues2 = predict_gender(inputfile)
    print('================end=================\n')

    # for row in predictedValues.iterrows():
    #     writeXML(outputfile, row[1]['userid'], row[1]['age'], row[1]['gender'], row[1]['ext'], row[1]['neu'],
    #              row[1]['agr'], row[1]['con'], row[1]['ope'])

    for index, row in relation_iris_frame.iterrows():
        # for age, use RelationPredictor_Final
        print(type(predictedValues[predictedValues['userid'] == row['userid']]))
        print(type(predictedValues[predictedValues['userid'] == row['userid']]['age']))
        print(type(predictedValues[predictedValues['userid'] == row['userid']]['age'].values))
        finaloutput[0] = predictedValues[predictedValues['userid'] == row['userid']]['age'].values[0]

        # for gender, compute majority among RelationPredictor, imageCNN1 and Prediction_Manish
        gender1 = image_CNN1_frame[image_CNN1_frame['userid'] == row['userid']]['gender'].values[0]
        if gender1 == 1:
            gender1 = 0
        elif gender1 == 0:
            gender1 = 1

        gender2 = predictedValues[predictedValues['userid'] == row['userid']]['gender'].values[0]
        if gender2 == 'male':
            gender2 = 0
        elif gender2 == 'female':
            gender2 = 1

        gender3 = predictedValues2[predictedValues2['userid'] == row['userid']]['gender'].values[0]
        if gender3 == 'male':
            gender3 = 0
        elif gender3 == 'female':
            gender3 = 1

        gender = float(gender1 + gender2 + gender3) / 3.
        gender_helper(gender)
        print('gender prediction for 3 models are:',gender1, gender2, gender3, '\n')
        print('ensembled gender is:', finaloutput[1], '\n')

        #for other compute avg of Relation_Iris and RelationPrediction_Final
        if row['age'] == '-':
            print('only on relationPrediction\n')
            finaloutput[2] = predictedValues[predictedValues['userid'] == row['userid']]['ope'].values[0]
            finaloutput[3] = predictedValues[predictedValues['userid'] == row['userid']]['con'].values[0]
            finaloutput[4] = predictedValues[predictedValues['userid'] == row['userid']]['ext'].values[0]
            finaloutput[5] = predictedValues[predictedValues['userid'] == row['userid']]['agr'].values[0]
            finaloutput[6] = predictedValues[predictedValues['userid'] == row['userid']]['neu'].values[0]
            print(finaloutput[2], finaloutput[3],finaloutput[4],finaloutput[5],finaloutput[6],'\n')

        else:
            print('on avg\n')
            finaloutput[2] = str((float(row['ope']) + float(predictedValues[predictedValues['userid'] == row['userid']]['ope'].values[0])) / 2.)
            finaloutput[3] = str((float(row['con']) + float(predictedValues[predictedValues['userid'] == row['userid']]['con'].values[0])) / 2.)
            finaloutput[4] = str((float(row['ext']) + float(predictedValues[predictedValues['userid'] == row['userid']]['ext'].values[0])) / 2.)
            finaloutput[5] = str((float(row['agr']) + float(predictedValues[predictedValues['userid'] == row['userid']]['agr'].values[0])) / 2.)
            finaloutput[6] = str((float(row['neu']) + float(predictedValues[predictedValues['userid'] == row['userid']]['neu'].values[0])) / 2.)
            print(finaloutput[2], finaloutput[3], finaloutput[4], finaloutput[5], finaloutput[6],'\n')

        output_frame.ix[index, 'age'] = finaloutput[0]
        output_frame.ix[index, 'gender'] = finaloutput[1]
        output_frame.ix[index, 'ope'] = finaloutput[2]
        output_frame.ix[index, 'con'] = finaloutput[3]
        output_frame.ix[index, 'ext'] = finaloutput[4]
        output_frame.ix[index, 'agr'] = finaloutput[5]
        output_frame.ix[index, 'neu'] = finaloutput[6]
        
        writeXML(outputfile, row['userid'],finaloutput)

# -----------------------------------
# Method to get the output dataframe
# -----------------------------------
def getOutput():
    global output_frame
    return output_frame

# -----------------------------------------------------------
# Helper method to write the output .xml file
#
# @param file: string, output file path
# @param userid: string, current user id
# @param output: list, prediction result for the current user
# -----------------------------------------------------------
def writeXML(file, userid, output):

    newFile = open(file+userid+'.xml','w')
    newFile.write('<user\nid="' + userid + '"\n' + 'age_group="' + str(finaloutput[0]) + '"\n' + 'gender="' + str(finaloutput[1])
                  + '"\n' + 'extrovert="' + str(finaloutput[4]) + '"\n' + 'neurotic="' + str(finaloutput[6]) + '"\n' + 'agreeable="'
                  + str(finaloutput[5]) + '"\n' + 'conscientious="' + str(finaloutput[3]) + '"\n' + 'open="' + str(finaloutput[2])
                  + '"\n' + '/>')
    newFile.close()





