# Cell2cell - ML class project
"""This python code is used for modeling part of customer churn data analysis. 
Majority of exploratory data analysis and some feature engineering tasks
has been done using Excel."""

## Importing the libraries
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import statsmodels.formula.api as smf
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder


## Importing the dataset
dataset = pd.read_csv('cell2celltrainL2.csv')
datasetnew = dataset.drop(['CallForwardingCalls', 'ServiceArea', 'CustomerID', 'MonthsInService', 'CED1', 'CED2', 'CED3', 'CED4'], axis=1)

## Encoding categorical data
"""selected 16 largest areas only. Decided to encode manually"""
datasetnew['NYC'] = np.where(datasetnew['CityCode'].str.contains('NYC'), 1, 0)
datasetnew['LAX'] = np.where(datasetnew['CityCode'].str.contains('LAX'), 1, 0)
datasetnew['SFR'] = np.where(datasetnew['CityCode'].str.contains('SFR'), 1, 0)
datasetnew['APC'] = np.where(datasetnew['CityCode'].str.contains('APC'), 1, 0)
datasetnew['DAL'] = np.where(datasetnew['CityCode'].str.contains('DAL'), 1, 0)
datasetnew['SAN'] = np.where(datasetnew['CityCode'].str.contains('SAN'), 1, 0)
datasetnew['CHI'] = np.where(datasetnew['CityCode'].str.contains('CHI'), 1, 0)
datasetnew['FLN'] = np.where(datasetnew['CityCode'].str.contains('FLN'), 1, 0)
datasetnew['MIA'] = np.where(datasetnew['CityCode'].str.contains('MIA'), 1, 0)
datasetnew['ATL'] = np.where(datasetnew['CityCode'].str.contains('ATL'), 1, 0)
datasetnew['OHI'] = np.where(datasetnew['CityCode'].str.contains('OHI'), 1, 0)
datasetnew['HOU'] = np.where(datasetnew['CityCode'].str.contains('HOU'), 1, 0)
datasetnew['NCR'] = np.where(datasetnew['CityCode'].str.contains('NCR'), 1, 0)
datasetnew['NEV'] = np.where(datasetnew['CityCode'].str.contains('NEV'), 1, 0)
datasetnew['DET'] = np.where(datasetnew['CityCode'].str.contains('DET'), 1, 0)
datasetnew['BOS'] = np.where(datasetnew['CityCode'].str.contains('BOS'), 1, 0)

datasetnew['Professional'] = np.where(datasetnew['Occupation'].str.contains('Professional'), 1, 0)
datasetnew['Crafts'] = np.where(datasetnew['Occupation'].str.contains('Crafts'), 1, 0)
datasetnew['Other'] = np.where(datasetnew['Occupation'].str.contains('Other'), 1, 0)

datasetnew['Suburban'] = np.where(datasetnew['PrizmCode'].str.contains('Suburban'), 1, 0)
datasetnew['Town'] = np.where(datasetnew['PrizmCode'].str.contains('Town'), 1, 0)
datasetnew['Rural'] = np.where(datasetnew['PrizmCode'].str.contains('Rural'), 1, 0)

datasetnew = datasetnew.drop(['CityCode', 'Occupation', 'PrizmCode'], axis=1)

datasetnew['Churn']=datasetnew['Churn'].map({'Yes':1,'No':0})
datasetnew['ChildrenInHH']=datasetnew['ChildrenInHH'].map({'Yes':1,'No':0})
datasetnew['HandsetRefurbished']=datasetnew['HandsetRefurbished'].map({'Yes':1,'No':0})
datasetnew['HandsetWebCapable']=datasetnew['HandsetWebCapable'].map({'Yes':1,'No':0})
datasetnew['TruckOwner']=datasetnew['TruckOwner'].map({'Yes':1,'No':0})
datasetnew['RVOwner']=datasetnew['RVOwner'].map({'Yes':1,'No':0})
datasetnew['Homeownership']=datasetnew['Homeownership'].map({'Yes':1,'No':0})
datasetnew['RespondsToMailOffers']=datasetnew['RespondsToMailOffers'].map({'Yes':1,'No':0})
datasetnew['OptOutMailings']=datasetnew['OptOutMailings'].map({'Yes':1,'No':0})
datasetnew['NonUSTravel']=datasetnew['NonUSTravel'].map({'Yes':1,'No':0})
datasetnew['OwnsComputer']=datasetnew['OwnsComputer'].map({'Yes':1,'No':0})
datasetnew['HasCreditCard']=datasetnew['HasCreditCard'].map({'Yes':1,'No':0})
datasetnew['NewCellphoneUser']=datasetnew['NewCellphoneUser'].map({'Yes':1,'No':0})
datasetnew['OwnsMotorcycle']=datasetnew['OwnsMotorcycle'].map({'Yes':1,'No':0})
datasetnew['MadeCallToRetentionTeam']=datasetnew['MadeCallToRetentionTeam'].map({'Yes':1,'No':0})
datasetnew['MaritalStatus']=datasetnew['MaritalStatus'].map({'Yes':1,'No':0})
datasetnew['NotNewCellphoneUser']=datasetnew['NotNewCellphoneUser'].map({'Yes':1,'No':0})
datasetnew['BuysViaMailOrder']=datasetnew['BuysViaMailOrder'].map({'Yes':1,'No':0})

# Taking care of missing data
datasetnew = datasetnew.fillna(datasetnew.mean())

# adding square or ABS terms to some variable
datasetnew['SQMonthlyRevenue'] = datasetnew['MonthlyRevenue'] * datasetnew['MonthlyRevenue']/1000
datasetnew['SQMonthlyMinutes'] = datasetnew['MonthlyMinutes'] * datasetnew['MonthlyMinutes']/1000
datasetnew['SQTotalRecurringCharge'] = datasetnew['TotalRecurringCharge'] * datasetnew['TotalRecurringCharge']/1000
# datasetnew['SQPercChangeMinutes'] = datasetnew['PercChangeMinutes'] * datasetnew['PercChangeMinutes']/1000
# datasetnew['SQPercChangeRevenues'] = datasetnew['PercChangeRevenues'] * datasetnew['PercChangeRevenues']/1000

datasetnew['ABSPercChangeMinutes'] = datasetnew['PercChangeMinutes'].abs()
datasetnew['ABSPercChangeRevenues'] = datasetnew['PercChangeRevenues'].abs()

features=list(datasetnew.columns.drop(['Churn']))

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split  #this has been deprecated
from sklearn.model_selection import train_test_split  #Updated
data_train, data_test = train_test_split(datasetnew, test_size=0.2, random_state = 0)

## Feature Scaling
"""Depending on the model to be used, it might or might not applied"""
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler()
# data_train = sc.fit_transform(data_train)
# data_test = sc.transform(data_test)

# data_train = pd.DataFrame(sc.fit_transform(data_train),columns = data_train.columns)
# data_test = pd.DataFrame(sc.fit_transform(data_test),columns = data_test.columns)

##Remoring features which are less significant
"""stage 1, if use two stage screening. could skip stage 1 and go to final stage directly"""
# univariate testing. Remove variables with pvalue>0.10
features_tmp=features[:]
# for s in features:
#    f_unit = 'Churn ~ ' + s
#    lregu = smf.logit(formula = f_unit, data = data_train).fit()
##    print(lregu.summary())
#    if lregu.pvalues[1]>0.10:
#        features_tmp.remove(s)  #remove feature s from the list
#        print('Feature: "'+s+'" removed')
# features_tmp now includes only variables with pvalue<=0.10

# tempory model after removing variables from unit variate testing
f_tmp='Churn~'+'+'.join(features_tmp)
lregt = smf.logit(formula = f_tmp, data = data_train).fit()
print(lregt.summary())

##Stepwise removal.  Remove variable with highest pvalue>0.05 in each iteration.
"""final stage of screening - stepwise backward elimination"""

pvalue_max=max(lregt.pvalues.drop('Intercept'))
while pvalue_max>0.05:
    feat_max=lregt.pvalues.drop('Intercept').idxmax()
    features_tmp.remove(feat_max)
    f_tmp='Churn~'+'+'.join(features_tmp)
    lregt = smf.logit(formula = f_tmp, data = data_train).fit()
    print(lregt.summary())
    pvalue_max=max(lregt.pvalues.drop('Intercept'))
    input('Removed feature "'+feat_max+'". Press enter to continue...')
    
print('Final feature list:', features_tmp)    

#optimized model after removing top non-predictors, stepwise Backward elimination
f_Opt='Churn~'+'+'.join(features_tmp)
lregO = smf.logit(formula = f_Opt, data = data_train).fit()
print(lregO.summary())

#Print Odds Ratio
np.exp(lregO.params)

#Predict the test sample / validation sample
PredictResults=lregO.predict(data_test)

#What the results look like
print(PredictResults[0:10])

#Draw a histogram for the predicted probabilities
# import matplotlib.pyplot as plt
pr=PredictResults.to_frame()
pr.columns=['a']

pr.a.hist()
plt.title('Histogram of Probability')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.savefig('hist_prob')

##to transform probability to churn or not churn
predictions_nominal = [ 1 if x >= 0.50 else 0 for x in PredictResults]

##Compare the actual and the prediction to get the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(data_test["Churn"], predictions_nominal)
print(cm)
 

##Explore various other models
"""
Explored various other models: 
1. Random Forest
2. kernel SVM
3. KNN
"""
  
# Prepare the training data
X_train = data_train.loc[:, features_tmp].values
y_train = data_train.iloc[:, 0].values
X_test = data_test.loc[:, features_tmp].values
y_test = data_test.iloc[:, 0].values
   
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 1. Random Forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# 2. kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# 3. KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Fitting classifier to the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)

