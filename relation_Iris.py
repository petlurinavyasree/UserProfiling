#!/usr/bin/env python3
# course: TCSS555
# User profile in social media - relation: page score
# date: 10/10/2017
# name: Team 4 - Iris Xia
# description: Python file to generate prediction results using page score methods on relation data.
import pandas as pd

profile_test = pd.DataFrame()

# ----------------------------------------------------------------------
# Method for generating prediction results from user data from inputfile.
#
# @param inputfile: string, input file path
# ----------------------------------------------------------------------
def get_result_relation(inputfile):
    test_relation_csv = pd.read_csv(inputfile + "relation/relation.csv", index_col = 0)
    relation_test = pd.DataFrame(test_relation_csv,columns = ['userid','like_id'])
    test_profile_csv = pd.read_csv(inputfile + "profile/profile.csv", index_col = 0)
    profile_test = pd.DataFrame(test_profile_csv,columns=['userid','age','gender','ope','con','ext','agr','neu'])
    return relation_result(relation_test, profile_test)

# ------------------------------------------------------------------------
# Helper Method for generating prediction results given relation test data
# for users in the test profile.
#
# @param relation_test: pd.dataframe dataframe for test relation data
# @param profile_test: pd.dataframe dataframe for test profile
# ------------------------------------------------------------------------
def relation_result(relation_test, profile_test):
    relation_csv = pd.read_csv("/data/training/relation/relation.csv", index_col = 0)
    #relation_csv = pd.read_csv("training/relation/relation.csv", index_col=0)
    relation_train = pd.DataFrame(relation_csv,columns = ['userid','like_id'])
    profile_csv = pd.read_csv("/data/training/profile/profile.csv", index_col = 0)
    #profile_csv = pd.read_csv("training/profile/profile.csv", index_col=0)
    profile_train= pd.DataFrame(profile_csv,columns = ['userid','age','gender','ope','con','ext','agr','neu'])
    return test_relation_result(relation_train, relation_test, profile_train, profile_test)


# ------------------------------------------------------------------------
# Helper Method for generating prediction results given relation test data
# for users in the test profile based on training relation data.
#
# @param relation_test: pd.dataframe dataframe for test relation data
# @param profile_test: pd.dataframe dataframe for test profile
# @param relation_train: pd.dataframe dataframe for train relation data
# @param profile_train: pd.dataframe dataframe for train profile
# @return profile_test: pd.dataframe datafraome for prediction results
# ------------------------------------------------------------------------
def test_relation_result(relation_train, relation_test, profile_train, profile_test):
    page_score_csv = pd.read_csv("page_score.csv")
    page_score = pd.DataFrame(page_score_csv, columns = ['like_id', 'age','gender','ope','con','ext','agr','neu'])
    for index, row in profile_test.iterrows():
        # find all like_id for current userid
        print("Dealling with ",index, "th user with userid: ", row['userid'], '\n')
        thisUserLikeId = relation_test[relation_test['userid'] == row['userid']]
        print('this user likes ', len(thisUserLikeId), ' pages\n')
        thisUserScore = pd.merge(thisUserLikeId, page_score, on=['like_id'])
        num = len(thisUserScore)
        print(num, "pages are learned\n")
        # for index2, row2 in thisUserLikeId.iterrows():
        #     #thisPageScore = page_score[page_score['like_id'] == row2['like_id']]
        #     if len(thisPageScore) != 0:
        #         count += 1
        #         age += thisPageScore['age'].values[0]
        #         gender += thisPageScore['gender'].values[0]
        #         ope += thisPageScore['ope'].values[0]
        #         con += thisPageScore['con'].values[0]
        #         ext += thisPageScore['ext'].values[0]
        #         agr += thisPageScore['agr'].values[0]
        #         neu += thisPageScore['neu'].values[0]


        if num != 0:
            profile_test.ix[index, 'age'] = thisUserScore['age'].mean()
            profile_test.ix[index, 'gender'] = thisUserScore['gender'].mean()
            profile_test.ix[index, 'ope'] = thisUserScore['ope'].mean()
            profile_test.ix[index, 'con'] = thisUserScore['con'].mean()
            profile_test.ix[index, 'ext'] = thisUserScore['ext'].mean()
            profile_test.ix[index, 'agr'] = thisUserScore['agr'].mean()
            profile_test.ix[index, 'neu'] = thisUserScore['neu'].mean()
            #print("Age, gender, ope, con, ext, agr, neu:", float(age / count), float(gender / count), float(ope / count),float(con / count), float(ext / count), float(agr / count), float(neu / count), "\n")
            # print(test_profile_frame.ix[index,'age'])
            # profile_test.ix[index, 'age'] = float(age / count)
            # # print(float(age/count))
            # profile_test.ix[index, 'gender'] = float(gender / count)
            # profile_test.ix[index, 'ope'] = float(ope / count)
            # profile_test.ix[index, 'con'] = float(con / count)
            # profile_test.ix[index, 'ext'] = float(ext / count)
            # profile_test.ix[index, 'agr'] = float(agr / count)
            # profile_test.ix[index, 'neu'] = float(neu / count)

    # profile_test.to_csv("relation_Irishahaha.csv")
    return profile_test
