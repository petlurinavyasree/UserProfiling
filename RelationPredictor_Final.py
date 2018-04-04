import pandas as pd
import numpy as np
from  sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from math import sqrt
from collections import Counter
from sklearn import decomposition
from sklearn import linear_model
from sklearn.model_selection import KFold

class RelationPredictor:
    # -------------------------------------------------------------
    # Helper method fof Kfold validation
    # @param X:training data
    # @param labelType :string,Age/gendery/personality
    # @param label1 : value of label
    # @param label2 : value of label
    # @param label3 : value of label
    # @param label4 : value of label
    # @param label5 : value of label
    # -------------------------------------------------------------
    def kFoldValidation(self, X, labelType, label1, label2='', label3='', label4='', label5=''):
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]

            if labelType == 'personality':
                opey_train, opey_test = label1[train_index], label1[test_index]
                cony_train, cony_test = label2[train_index], label2[test_index]
                exty_train, exty_test = label3[train_index], label3[test_index]
                agry_train, agry_test = label4[train_index], label4[test_index]
                neuy_train, neuy_test = label5[train_index], label5[test_index]

                model = linear_model.LinearRegression(normalize=True)
                model.fit(X_train, opey_train)
                opey_predicted = model.predict(X_test)
                print(sqrt(mean_squared_error(opey_test, opey_predicted)))

                model.fit(X_train, cony_train)
                cony_predicted = model.predict(X_test)
                print(sqrt(mean_squared_error(cony_test, cony_predicted)))

                model.fit(X_train, exty_train)
                exty_predicted = model.predict(X_test)
                print(sqrt(mean_squared_error(exty_test, exty_predicted)))

                model.fit(X_train, agry_train)
                agry_predicted = model.predict(X_test)
                print(sqrt(mean_squared_error(agry_test, agry_predicted)))

                model.fit(X_train, neuy_train)
                neuy_predicted = model.predict(X_test)
                print(sqrt(mean_squared_error(neuy_test, neuy_predicted)))
                print('##################')
            else:
                y_train, y_test = label1[train_index], label1[test_index]
                logisticModel = linear_model.LogisticRegression()
                logisticModel.fit(X_train, y_train)
                y_predicted = logisticModel.predict(X_test)
                print(accuracy_score(y_test, y_predicted))
    # -------------------------------------------------------------
    # Helper method to predict Gender
    #
    # @param trainingData
    # @param testingData
    #@param doKFold : flag to do kfold
    # -------------------------------------------------------------
    def predictGender(self, trainingData, testingData, doKFold=True):

        #Get Matrix for Dimensionality Reduction..
        X = trainingData.ix[:, 1:-8]

        # Get gender labels as y..
        genderY = trainingData.ix[:, 'gender']

        if doKFold == True:
            RelationPredictor.kFoldValidation(X, labelType='gender', label1=genderY)
        else:
            X_test = testingData.ix[:, 1:-8]

            logisticModel = linear_model.LogisticRegression()
            logisticModel.fit(X, genderY)
            y_predicted = logisticModel.predict(X_test)
            y_predicted = pd.Series(y_predicted).map({0: 'male', 1: 'female'})
            return y_predicted
    # -------------------------------------------------------------
    # Helper method to predict Age
    #
    # @param trainingData
    # @param testingData
    #@param doKFold : flag to do kfold
    # -------------------------------------------------------------
    def predictAge(self, trainingData, testingData, doKFold=True):

        X = trainingData.ix[:, 1:-8]
        # Get age label as y..
        ageY = trainingData.ix[:, 'age']

        # Divide Age into 4 bins..(We have age 112 lol!)
        ageY = pd.cut(ageY, [0.0,24.0,34.0,49.0,1000.0], labels=[0,1,2,3], retbins=False, include_lowest=True)

        if doKFold == True:
            RelationPredictor.kFoldValidation(X, labelType='age', label1=ageY)
        else:
            X_test = testingData.ix[:, 1:-8]

            logisticModel = linear_model.LogisticRegression()
            logisticModel.fit(X, ageY)
            y_predicted = logisticModel.predict(X_test)

            # Convert age to buckets and return..
            y_predicted = pd.Series(y_predicted).map({0:'xx-24',1:'25-34',2:'35-49',3:'50-xx'})
            return y_predicted
    # -------------------------------------------------------------
    # Helper method to predict personlaity scores
    #
    # @param trainingData
    # @param testingData
    #@param doKFold : flag to do kfold
    # -------------------------------------------------------------
    def predictPersonalityScores(self, trainingData, testingData, doKFold=True):
        X = trainingData.ix[:, 1:-8]

        # Get personality labels as y..
        opeY = trainingData.ix[:, 'ope']
        conY = trainingData.ix[:, 'con']
        extY = trainingData.ix[:, 'ext']
        agrY = trainingData.ix[:, 'agr']
        neuY = trainingData.ix[:, 'neu']

        if doKFold == True:
            RelationPredictor.kFoldValidation(X, labelType='personality', label1=opeY, label2=conY, label3=extY, label4=agrY, label5=neuY)
        else:
            X_test = testingData.ix[:, 1:-8]

            model = linear_model.LinearRegression(normalize=True)
            model.fit(X, opeY)
            opey_predicted = model.predict(X_test)

            model.fit(X, conY)
            cony_predicted = model.predict(X_test)

            model.fit(X, extY)
            exty_predicted = model.predict(X_test)

            model.fit(X, agrY)
            agry_predicted = model.predict(X_test)

            model.fit(X, neuY)
            neuy_predicted = model.predict(X_test)

            return opey_predicted, cony_predicted, exty_predicted, agry_predicted, neuy_predicted
    # -------------------------------------------------------------
    # Helper method preprocessing the given data
    #@param inputDir: input directory path
    # -------------------------------------------------------------
    def relationPrediction(self, inputDir):
        print("Model may take atleast 20 mins for predictions..")
        # Read the user-like Data..
        relationDataFrame = pd.read_csv("/data/training/relation/relation.csv")

        # Get Count of No.of likes to each page..
        counts = relationDataFrame['like_id'].value_counts()

        # Filter out the least liked and Most liked pages..
        filteredFrame = relationDataFrame[relationDataFrame['like_id'].isin(counts[counts > 25].index)]
        filteredFrame = filteredFrame[filteredFrame['like_id'].isin(counts[counts < 1000].index)]

        # Read the test data..
        relationTestDataFrame = pd.read_csv(inputDir + "/relation/relation.csv")

        # Get unique like id's from test data..
        uniqueLikeIds = relationTestDataFrame.like_id.unique()

        filteredFrame = filteredFrame[filteredFrame.like_id.isin(uniqueLikeIds)]

        tempTestDataFrame = relationTestDataFrame[relationTestDataFrame.like_id.isin(filteredFrame.like_id.unique())]

        uselessTestUsers = relationTestDataFrame[~relationTestDataFrame.userid.isin(tempTestDataFrame.userid.unique())]
        print(len(uselessTestUsers.userid.unique()))
        uselessTestUsers = uselessTestUsers[~uselessTestUsers.duplicated(subset='userid')]
        tempTestDataFrame = tempTestDataFrame.append(uselessTestUsers)

        relationTestDataFrame = tempTestDataFrame

        # append test data to train data..
        filteredFrame = filteredFrame.append(relationTestDataFrame)
        print(filteredFrame.shape)

        #  Get a user-like Cross Matrix..
        crossedTable = pd.crosstab(filteredFrame['userid'], filteredFrame['like_id'])
        crossedFrameWithIndex = pd.DataFrame(crossedTable.to_records())

        # Apply PCA to the data..
        X = crossedFrameWithIndex.ix[:, 1:]
        pca = decomposition.TruncatedSVD(n_components=100)
        X = pca.fit_transform(X)

        pcaedDataFrame = pd.DataFrame(X, index=crossedFrameWithIndex.ix[:,'userid'])
        pcaedDataFrame.reset_index(level=0, inplace=True)

        # Seperate test data from training data..
        trainDataFrame = pcaedDataFrame[~pcaedDataFrame.userid.isin(relationTestDataFrame.userid.unique())]
        testDataFrame = pcaedDataFrame[pcaedDataFrame.userid.isin(relationTestDataFrame.userid.unique())]

        # Merge the test userid's and training userid's with their Profile..
        trainProfileDataFrame = pd.read_csv("/data/training/profile/profile.csv")
        testProfileDataFrame = pd.read_csv(inputDir + "/profile/profile.csv")

        trainDataFrame = pd.merge(trainDataFrame, trainProfileDataFrame, on='userid')
        testDataFrame = pd.merge(testDataFrame, testProfileDataFrame, on='userid')

        gender = RelationPredictor.predictGender(self, trainingData=trainDataFrame, testingData=testDataFrame, doKFold=False)
        age = RelationPredictor.predictAge(self, trainingData=trainDataFrame, testingData=testDataFrame, doKFold=False)
        ope, con, ext, agr, neu = RelationPredictor.predictPersonalityScores(self, trainingData=trainDataFrame, testingData=testDataFrame, doKFold=False)


        finDataFrame = pd.DataFrame({'userid':testDataFrame['userid'], 'gender': gender, 'age': age, 'ope': ope, 'con': con, 'ext': ext, 'agr': agr, 'neu': neu})
        # print("Gender Acc: " + str(accuracy_score(testDataFrame.ix[:,'gender'], finDataFrame['gender'])))
        # testingAge = pd.cut(testDataFrame.ix[:,'age'], [0.0, 24.0, 34.0, 49.0, 1000.0], labels=[0, 1, 2, 3], retbins=False, include_lowest=True)
        # print("Age Acc: " + str(accuracy_score(testingAge, finDataFrame['age'])))
        # print("Ope RMSE: " + str(sqrt(mean_squared_error(testDataFrame.ix[:,'ope'], finDataFrame['ope']))))
        # print("Con RMSE: " + str(sqrt(mean_squared_error(testDataFrame.ix[:,'con'], finDataFrame['con']))))
        # print("Ext RMSE: " + str(sqrt(mean_squared_error(testDataFrame.ix[:,'ext'], finDataFrame['ext']))))
        # print("Agr RMSE: " + str(sqrt(mean_squared_error(testDataFrame.ix[:,'agr'], finDataFrame['agr']))))
        # print("Neu RMSE: " + str(sqrt(mean_squared_error(testDataFrame.ix[:,'neu'], finDataFrame['neu']))))
        return finDataFrame
