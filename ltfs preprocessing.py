#PREPROCESSING

#CONVERT STRS TO FLAGS
#REMOVE UNIQUEID COL
#CONVERT DATE TO AGE
#COONVRT ELAPSED TIME TO NANOSECONDS
########################################################################
#RESOURCES/LINKS
########################################################################
#https://www.quora.com/What-are-good-ways-to-handle-discrete-and-continuous-inputs-together
#https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
#https://stats.stackexchange.com/questions/26764/predicting-with-both-continuous-and-categorical-features
#https://stats.stackexchange.com/questions/364088/how-to-best-code-the-n-a-response-of-the-likert-type-rating-scale
#https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
#https://towardsdatascience.com/predicting-loan-repayment-5df4e0023e92
#https://medium.com/@andrejusb/machine-learning-date-feature-transformation-explained-4feb774c9dbe
#https://stats.stackexchange.com/questions/105959/best-way-to-turn-a-date-into-a-numerical-feature
########################################################################
#IMPORTS
########################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
from datetime import datetime
########################################################################
#READ FILE, STATE columns with NULLS
########################################################################
def read_csv(filename):
	'''
	INPUTS
	filename: name of file

	OUTPUT
	df: pandas matrix of file
	-print out number of null values per column
	'''

	return pd.read_csv(filename, sep=',').drop(
		['UniqueID'], axis=1)

########################################################################
#ELIMINATE ROWS WITH NULL VALUES
########################################################################
def eliminate_NULL(df, name):
	df.dropna(subset=[name], inplace=True)
	print("Sum of null values in each feature:\n")
	print(f"{df.isnull().sum()}")
	df.head()

########################################################################
#CONVERT DATE STAMPS TO NANOSECONDS 04/21/19 12AM: 1555804800000000000ns
########################################################################
def date_to_age(df, name, new_name, datatype='year'):
	'''
	INPUTS
	pd_col: column containing date values

	OUTPUT
	col: new column containing time in years
	'''

	df1 = pd.to_datetime(df[name], format='%d-%m-%y')
	col_today = pd.to_datetime('20190101')
	for i in range(len(df1.index)):
		if 2020 <= df1.iloc[i].year <= 2068:
			df1.iloc[i] = df1.iloc[i] - pd.DateOffset(years=100)
	if datatype is not 'year':
			df[new_name] = (col_today - df1).dt.days
	else:
		df[new_name] = round((col_today - df1).dt.days/365,1)#new_col
	df=df.drop([name], axis=1)
	return df

########################################################################
#CONVERT salary column of strings to flags
########################################################################
def salary_type(df, name):
	'''
	INPUTS
	df: dataframe of data

	OUTPUT
	(df) modified dataframe with one hot encoding for salary
	'''
	sal_le, sal_ohe = LabelEncoder(), OneHotEncoder()
	sal_labels = sal_le.fit_transform(df[name])
	sal_feature_arr = sal_ohe.fit_transform(sal_labels.reshape(-1,1)).toarray()
	sal_features = pd.DataFrame(sal_feature_arr,
		columns=list(sal_le.classes_), index=df.index)
	df = df.join(sal_features).drop([name], axis=1)
	return df
	
########################################################################
#CONVERT time elapsed columns into months
########################################################################
def time_elapsed(df, name):
	'''
	INPUTS
	pd_col: column containing elapsed time information as strings

	OUTPUT
	pd_col: same column now with values in terms of months
	'''
	df[name]=df[name].str.replace('yrs', '', regex=False)
	df[name]=df[name].str.replace('mon', '', regex=False)
	df2 = df[name].str.split(' ', expand=True).astype(int)
	df[name] = df2.iloc[:,0]*12 + df2.iloc[:,1]

########################################################################
#CONVERT BUREAU CREDIT RISK CATEGORIES INTO FLAGS
########################################################################
def credit_risk(df, name):
	'''
	INPUTS
	df: dataframe of data

	OUTPUT
	pd_col: same column now with values as flags...note the categories
	'''
	df[name]=df[name].str.replace('No Bureau History Available', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: Not Enough Info available on the customer', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: No Activity seen on the customer (Inactive)', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: No Updates available in last 36 months', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: Sufficient History Not Available', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: Only a Guarantor', 'Not Scored', regex=False)
	df[name]=df[name].str.replace('Not Scored: More than 50 active Accounts found', 'Not Scored', regex=False)    
	df.loc[df[name].str.contains('Very Low Risk'), name] = 'Very Low-Risk'
	df.loc[df[name].str.contains('Very High Risk'), name] = 'Very High-Risk'
	df.loc[df[name].str.contains('Low Risk'), name] = 'Low Risk'
	df.loc[df[name].str.contains('High Risk'), name] = 'High Risk'
	df.loc[df[name].str.contains('Medium Risk'), name] = 'Medium Risk'
	df[name] = df[name].map({'Very Low-Risk': 1, 'Low Risk': 2, 'Not Scored': 3, 
			   'Medium Risk': 4, 'High Risk': 5, 'Very High-Risk': 6})

########################################################################
#NORMALIZING DATASET
########################################################################
def pseudo_norm(X_data, mean, std): #normalize nominal X given the mean and std
	X_data = (X_data-mean)/std
	return X_data

def norm(X):
	avg = np.mean(X.values, axis = 0)
	stdev = np.std(X.values, axis = 0)
	X = (X-avg)/stdev
	print([i for i in range(stdev.shape[0]) if stdev[i]==0])

	np.savetxt('stats_avg.csv', avg, delimiter=',', fmt = '%f')
	np.savetxt('stats_std.csv', stdev, delimiter=',',fmt = '%f')
		
	print('normalized file and saved its stats')
	return X, avg, stdev
########################################################################
#CALLING FUNCTIONS TO MANIPULATE DATAFRAME
########################################################################

#read file
file = read_csv('train.csv')
eliminate_NULL(file, 'Employment.Type')
Y = file['loan_default']

#preprocessing training
file = file.drop(['loan_default'],axis=1)
file = file.drop(['MobileNo_Avl_Flag'], axis=1) #invariant
file = date_to_age(file, 'Date.of.Birth', 'AGE')
file = date_to_age(file, 'DisbursalDate', 'DAYS_DISBURSAL', datatype='days')
file = salary_type(file, 'Employment.Type')
credit_risk(file, 'PERFORM_CNS.SCORE.DESCRIPTION')
time_elapsed(file, 'AVERAGE.ACCT.AGE')
time_elapsed(file, 'CREDIT.HISTORY.LENGTH')

#separate encoded categorical from cts variables
names_non_flags = list(file)
flags = ['Aadhar_flag','PAN_flag','VoterID_flag', 'Driving_flag','Passport_flag','Salaried','Self employed']
for flag in flags:
	names_non_flags.remove(flag)

#normalize training
X, avg, stdev = norm(file[names_non_flags])
X = X.join(file[flags])

#read test file
test = read_csv('test-file.csv')

#preprocessing test
test = test.drop(['MobileNo_Avl_Flag'], axis=1) #invariant
test = date_to_age(test, 'Date.of.Birth', 'AGE')
test = date_to_age(test, 'DisbursalDate', 'DAYS_DISBURSAL', datatype='days')
test = salary_type(test, 'Employment.Type')
credit_risk(test, 'PERFORM_CNS.SCORE.DESCRIPTION')
time_elapsed(test, 'AVERAGE.ACCT.AGE')
time_elapsed(test, 'CREDIT.HISTORY.LENGTH')

#pseudo-normalize test
X_test = pseudo_norm(test[names_non_flags], avg, stdev)
X_test = X_test.join(test[flags])

#READY FOR FEATURE SELECTION!