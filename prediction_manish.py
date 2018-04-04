# Course: TCSS555 (Team 4)
# 11/12/2017
# Programmer: Manish KC
# Starter-code provided by: Dr. Martine De Cock
# Description: Naive Bayes model for gender and age prediction from transcripts of 9500 facebook users 




#import nltk
import codecs
import os
import glob
import string
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
#from nltk.tokenize import word_tokenize




def predict_gender(path_to_test):
 		# Reads "profile.csv" and add "age-group" column to the new dataframe 
		#Change location of file accordingly
		
		df = pd.read_csv("/data/training/profile/profile.csv")
		df_profile = df.loc[:,['userid', 'gender', 'age']]
		df_profile['age-group'] = np.where(df['age']<=24, 'xx-24',
							  np.where(df['age']<=34, '25-34',
							  np.where(df['age']<=49, '35-49',
							  np.where(df['age']>=50, '50-xx',	'NONE'))))

		df_profile.drop('age', axis=1, inplace=True)
		#print (df_profile)  


		#Changes directory where text files are located and create a dataframe of the filenames 
		#Change location of file accordingly
		path = '/data/training/text'
		filelist = os.listdir(path)
		df_textFiles = pd.DataFrame()
		df_textFiles = df_textFiles.append(pd.DataFrame(filelist, columns=['userid']), ignore_index=True)
		df_textFiles['userid'] = df_textFiles['userid'].map(lambda x: str(x)[:-4])
		#print (df_textFiles)


		#Reads contents of each text file(ignores unicode error) and creates "new_df" to store all the texts
		text = []
		#words = set(nltk.corpus.words.words())


		for files in glob.glob(os.path.join(path, "*.txt")):
			with codecs.open (files, 'r', encoding = 'utf-8', errors='ignore') as f:
				#new= " ".join(w for w in nltk.wordpunct_tokenize(f.read())  if w.lower() in words)
				new = f.read()
				text.append(new)
								



		df_all_texts = pd.DataFrame()
		df_all_texts = df_all_texts.append(pd.DataFrame(text, columns =['transcripts']), ignore_index=True)
		new_df = df_textFiles.join(df_all_texts)


		#Creates a final dataframe "final_df" merging based on "userid"
		final_df = pd.merge(df_profile, new_df, on='userid') 
		final_df['gender'] = np.where(final_df['gender']==0.0, 'male', 'female')
		#print (final_df)
		print ("loading...")
		
		
		########################################Test Only #################################################
		 		# Reads "profile.csv" and add "age-group" column to the new dataframe 
		#Change location of file accordingly
		print ("ok")
		
		os.chdir(path_to_test)
		print (path_to_test)
		'''
		#os.chdir("/home/itadmin")  # Change this 
		df = pd.read_csv("profile/profile.csv")
		df_profile_ = df.loc[:,['userid', 'gender', 'age']]
				
		df_profile['age-group'] = np.where(df['age']<=24, 'xx-24',
							  np.where(df['age']<=34, '25-34',
							  np.where(df['age']<=49, '35-49',
							  np.where(df['age']>=50, '50-xx',	'NONE'))))

		
		df_profile.drop('age', axis=1, inplace=True)
		print (df_profile)  
		'''


		#Changes directory where text files are located and create a dataframe of the filenames 
		#Change location of file accordingly

		path = path_to_test 
		path = 'text'
		filelist = os.listdir(path)
		df_textFiles = pd.DataFrame()
		df_textFiles = df_textFiles.append(pd.DataFrame(filelist, columns=['userid']), ignore_index=True)
		df_textFiles['userid'] = df_textFiles['userid'].map(lambda x: str(x)[:-4])
		#print (df_textFiles)


		#Reads contents of each text file(ignores unicode error) and creates "new_df" to store all the texts
		text = []
		#words = set(nltk.corpus.words.words())


		for files in glob.glob(os.path.join(path, "*.txt")):
			with codecs.open (files, 'r', encoding = 'utf-8', errors='ignore') as f:
				#new= " ".join(w for w in nltk.wordpunct_tokenize(f.read())  if w.lower() in words)
				new = f.read()				
				text.append(new)

								



		df_all_texts = pd.DataFrame()
		df_all_texts = df_all_texts.append(pd.DataFrame(text, columns =['transcripts']), ignore_index=True)
		#new_df = df_textFiles.join(df_all_texts)
		final_df_test = df_textFiles.join(df_all_texts)


		#Creates a final dataframe "final_df" merging based on "userid"
		#final_df_test = pd.merge(df_profile, new_df, on='userid') 
		#final_df_test['gender'] = np.where(final_df_test['gender']==0.0, 'male', 'female')
		#print (final_df_test)
		
		
		
		
		######################################## Test End ###################################################
		
		


		# Splitting the data into 8500 training instances and 1000 test instances
		all_Ids_train = np.arange(len(final_df)) 
		all_Ids_test = np.arange(len(final_df_test))  		
		#random.shuffle(all_Ids)
		test_Ids = all_Ids_test[0:]
		train_Ids = all_Ids_train[0:]
		data_test = final_df_test.loc[test_Ids, :]
		data_train = final_df.loc[train_Ids, :]   
		#print (data_test)
		#print (data_train)

		

		# Training a Naive Bayes model
		count_vect = CountVectorizer()
		X_train = count_vect.fit_transform(data_train['transcripts'])
		y_train = data_train['gender']
		clf = MultinomialNB()
		clf.fit(X_train, data_train['gender'])

		# Testing the Naive Bayes model
		X_test = count_vect.transform(data_test['transcripts'])
		#y_test = data_test['gender']
		y_predicted = clf.predict(X_test)

		#print("Accuracy for gender: %.2f" % accuracy_score(y_test,y_predicted))
		#print (y_predicted)
		

		
		xml_df = pd.DataFrame()
		xml_df_final = pd.DataFrame()
		xml_df_final = pd.concat([final_df_test['userid']], axis =1, keys=['userid'])
		xml_df = xml_df.append(pd.DataFrame(y_predicted, columns =['gender']), ignore_index=True)
		
		xml_df_final = xml_df_final.join(xml_df)
		xml_df_final['age_group'] = 'xx-24'
		xml_df_final['extrovert'] = '3.48'
		xml_df_final['neurotic'] = '2.89'
		xml_df_final['agreeable'] = '3.55'
		xml_df_final['conscientious'] = '3.25'
		xml_df_final['open'] = '3.94'
		
		
		#print (xml_df_final)
		print ("xml files have been saved")
		return xml_df_final
	


