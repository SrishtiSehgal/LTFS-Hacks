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
#https://machinelearningmastery.com/feature-selection-machine-learning-python/
#https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
#https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
########################################################################
#IMPORTS
########################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

########################################################################
#READ FILE, STATE columns with NULLS
########################################################################
def read_csv(filename):
	'''
	INPUTS
	filename: name of file

	OUTPUT
	df: pandas matrix of file
	'''

	file = pd.read_csv(filename, sep=',')
#	IDs = file['UniqueID']
#	file = file.drop(
#		['UniqueID'], axis=1)
	return file

########################################################################
#ELIMINATE ROWS WITH NULL VALUES
########################################################################
def eliminate_NULL(df, name):
	'''
	INPUTS
	df: dataframe of data
	name: the column that contains NA values

	OUTPUT
	-modified df to remove rows that contained NA values
	-print out number of null values per column
	'''

	df.dropna(subset=[name], inplace=True)
	print("Sum of null values in each feature:\n")
	print(f"{df.isnull().sum()}")
	df.head()

########################################################################
#CONVERT DATE STAMPS TO NANOSECONDS 04/21/19 12AM: 1555804800000000000ns
########################################################################
def date_to_age(df, name, new_name, f=True, datatype='year'):
	'''
	INPUTS
	df: dataframe of data
	name: the column that contains date values
	new_name: new column name that will contain the modified date rep
	f: based on style of date formating. True = %d-%m-%y
	datatype: if date formatting is to be year or in days

	OUTPUT
	df: modifed df with new column containing new date rep for col-name
	'''

	col_today = pd.to_datetime('20190101')
	if f:
		df1 = pd.to_datetime(df[name], format='%d-%m-%y')
		for i in range(len(df1.index)):
			if 2020 <= df1.iloc[i].year <= 2068:
				df1.iloc[i] = df1.iloc[i] - pd.DateOffset(years=100)
	else:
			df1 = pd.to_datetime(df[name])
	
	if datatype is not 'year':
			df[new_name] = (col_today - df1).dt.days
	else:
		df[new_name] = round((col_today - df1).dt.days/365,1)#new_col
	return df

########################################################################
#CONVERT salary column of strings to flags
########################################################################
def salary_type(df, name):
	'''
	INPUTS
	df: dataframe of data
	name: the column that contains data to be encoded

	OUTPUT
	df: modified dataframe with one hot encoding
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
	''''
	INPUTS
	df: dataframe of data
	name: the column that contains date values
	
	OUTPUT
	-modifed df with new column containing new date rep for col-name
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
	name: the column that contains date values
	
	OUTPUT
	-modifed df to rewrite strings in a certain way, then create a label
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
#NORMALIZATION
########################################################################
def pseudo_norm(X_data, mean, std): #normalize nominal X given the mean and std
	'''
	INPUTS
	X_data: data to be pseduonormed
	mean: mean of data
	std: std of data

	OUTPUT
	X_data: psudonormed data
	'''
	X_data = (X_data-mean)/std
	return X_data

def norm(X):
	'''
	INPUTS
	X: data to be normalized

	OUTPUT
	X: psudonormed data
	mean: mean of data
	std: std of data
	'''
	avg = np.mean(X.values, axis = 0)
	stdev = np.std(X.values, axis = 0)
	X = (X-avg)/stdev
	print([i for i in range(stdev.shape[0]) if stdev[i]==0])

	np.savetxt('stats_avg.csv', avg, delimiter=',', fmt = '%f')
	np.savetxt('stats_std.csv', stdev, delimiter=',',fmt = '%f')
		
	return X, avg, stdev

#######################################################################################
# PCA
#######################################################################################
def determine_PCA(train_X):
	def	plot_feature_importance(PC_info, PC_eigenvector):
		index = np.arange(len(PC_eigenvector))
		bar_width = 0.35
		opacity = 0.4
		fig, ax = plt.subplots(figsize=(10,8))
		rects1 = ax.bar(index, PC_eigenvector, 
						bar_width,
						alpha=opacity, 
						color='b',
						label='PCA')

		ax.set_xlabel('Features', fontsize=10)
		ax.set_ylabel('Eigenvalues', fontsize=10)
		ax.set_title('Importance of Features in Dataset\n'+PC_info)
		ax.set_xticks(index + bar_width / 2)
		ax.set_xticklabels(list(train_X), fontsize=10, rotation=90)
		ax.legend()
		fig.tight_layout()
		plt.show()
		plt.close()
	def plot_features(components):
		colors = plt.cm.viridis(np.linspace(0,1,components.shape[0]))
		fig, ax = plt.subplots(figsize=(20, 10))
		for i, component in enumerate(components):
			if i < 40:
				plt.plot(component, color = colors[i], label='Component no. '+str(i))
		ax.set_xticks(np.arange(components.shape[1]))
		ax.set_xticklabels(list(train_X), rotation=90)
		ax.set_xlabel('Features', fontsize=10)
		ax.set_ylabel('Eigenvalues', fontsize=10)
		plt.legend(fontsize='small')
		plt.show()
		
	for numPC in {21, 30}:
		pca = PCA(n_components=numPC)
		PCs = pca.fit(train_X.values)
		var_ratio = sorted(pca.explained_variance_ratio_, reverse=True)
		print(str(numPC) +'PCs retained '+str(sum(var_ratio)*100)+'% of information')
		for i in range(3):
			plot_feature_importance('PC at row {} \nPC variance: {}\n total variance: {}\n {} total PCs'.format(
				str(i),str(var_ratio[i]), str(sum(var_ratio)), str(numPC)), 
				abs(pca.components_[i,:]))

########################################################################
#DATASET STATS
########################################################################
def data_stats(target_data):
	'''
	INPUTS
	target_data: Y-values for the dataset
	OUTPUT
	-number of objects labelled as 1 (loan-defaulted)
	'''
	return len(target_data[target_data==1].index)

########################################################################
#CALLING FUNCTIONS TO MANIPULATE DATAFRAME
########################################################################
#read file
file = read_csv('train.csv')
eliminate_NULL(file, 'Employment.Type')

#preprocessing training
file = file.drop(['MobileNo_Avl_Flag'], axis=1) #invariant
file = date_to_age(file, 'Date.of.Birth', 'AGE')
file = date_to_age(file, 'DisbursalDate', 'DAYS_DISBURSAL', f=False, datatype='days')
#only data points where date of disbursement > DOB (cannot recieve loan if not born!)
file = file[file['DisbursalDate']>=file['Date.of.Birth']] 
file = file.drop(['Date.of.Birth'], axis=1)
file = file.drop(['DisbursalDate'], axis=1)
file = salary_type(file, 'Employment.Type')
Y = file['loan_default']
file = file.drop(['loan_default'],axis=1)
file = file.drop(['UniqueID'],axis=1)
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
eliminate_NULL(test, 'Employment.Type')

#preprocessing test
test = test.drop(['MobileNo_Avl_Flag'], axis=1) #invariant
test = date_to_age(test, 'Date.of.Birth', 'AGE')
test = date_to_age(test, 'DisbursalDate', 'DAYS_DISBURSAL', f=False, datatype='days')
#only data points where date of disbursement > DOB (cannot recieve loan if not born!)
test = test[test['DisbursalDate']>=test['Date.of.Birth']] 
test = test.drop(['Date.of.Birth'], axis=1)
test = test.drop(['DisbursalDate'], axis=1)
'''
the disbursement date is the date that a school credits a student’s account 
at the school or pays a student or parent borrower directly with Title IV 
funds received from the U.S. Department of Education (the Department) 
or with institutional funds in advance of receiving Title IV program funds.
'''
test = salary_type(test, 'Employment.Type')
credit_risk(test, 'PERFORM_CNS.SCORE.DESCRIPTION')
time_elapsed(test, 'AVERAGE.ACCT.AGE')
time_elapsed(test, 'CREDIT.HISTORY.LENGTH')
IDs = test['UniqueID']
test = test.drop(['UniqueID'],axis=1)
#pseudo-normalize test
X_test = pseudo_norm(test[names_non_flags], avg, stdev)
X_test = X_test.join(test[flags])

#READY FOR FEATURE SELECTION!
#determine_PCA(X)
#POST FEATURE SELECTION
X = X.drop(['Passport_flag'], axis=1)
X_test = X_test.drop(['Passport_flag'], axis=1)
X = X.drop(['Driving_flag'], axis=1)
X_test = X_test.drop(['Driving_flag'], axis=1)
X = X.drop(['VoterID_flag'], axis=1)
X_test = X_test.drop(['VoterID_flag'], axis=1)
X = X.drop(['PAN_flag'], axis=1)
X_test = X_test.drop(['PAN_flag'], axis=1)
X = X.drop(['Aadhar_flag'], axis=1)
X_test = X_test.drop(['Aadhar_flag'], axis=1)
X = X.drop(['Salaried'], axis=1)
X_test = X_test.drop(['Salaried'], axis=1)
X = X.drop(['Self employed'], axis=1)
X_test = X_test.drop(['Self employed'], axis=1) 
#maybe?
#X = X.drop(['DAYS_DISBURSAL'], axis=1)
#X_test = X_test.drop(['DAYS_DISBURSAL'], axis=1)
#X = X.drop(['Employee_code_ID'], axis=1)
#X_test = X_test.drop(['Employee_code_ID'], axis=1) 

# X.to_csv('trainingSet.csv', sep=',', encoding='utf-8')
# X_test.to_csv('testingSet.csv', sep=',', encoding='utf-8')

default_training = data_stats(Y)
    
import Classifiers as models

models.classification(X.values, Y.values, X_test.values, default_training,IDs)