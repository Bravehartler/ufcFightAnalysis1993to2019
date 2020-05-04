from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from  sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from  sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import model_selection
import rfpimp

import numpy as np
import pandas as pd
import plotly.express as px

import pickle

#load the previous generated cleanMLRDy File and the comparissons
df = pd.read_pickle("./data/clean/cleanMLRdy.pkl")
# make a set for blue wins
B_Winner_Blue = df.Winner_Red.copy()
B_Winner_Blue.loc[:] = 0
B_Winner_Blue[df.Winner_Red == df.Winner_Draw] = 1

# First let's calculate if red wins
df.drop(labels=["Winner_Draw"], axis=1, inplace = True) #=> drop draws
#Split the output and input data
X = df[df.columns[1:]]
y = df[df.columns[:1]]

#split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#use random wood ;) as classifier 
rfc = RandomForestClassifier()
#rfc.fit(X_train, y_train.values.ravel()) ==> instead of normal fitting, we use cross validation to estimate the parameters because test_recall was not so good after fitting

# check score and fit the rfc
rfc_scores = cross_validate(rfc, X, y.values.ravel(), cv=9, scoring=('accuracy', 'average_precision', 'recall','f1'), return_estimator=True)
model = rfc_scores['estimator']

#make a report on accuracy, precision, recall and f1
report = pd.DataFrame(index=list(rfc_scores.keys())[3:], columns=['Random Forest avg', 'Random Forest std', 'Chosen model params'])
for key in report.index:
    if(key=="test_recall"):
        #take the model closest do the mean
        checkval = np.abs(np.mean(rfc_scores[key])-rfc_scores[key])
        idx = np.where(np.min(checkval)==checkval)[0][0]
        rfc=model[idx]
        break


for key in report.index:
    report.loc[key] = [np.mean(rfc_scores[key]), np.std(rfc_scores[key]), rfc_scores[key][idx]]

report *= 100
report_R = report.astype(float).round(1)
rfc_R = rfc


#check importance of different features
models = rfc_scores['estimator']
imp_R = []
for model in models:
    imp_R.append(rfpimp.importances(model, X_test, y_test)[0:3]) # ==> check many features of the tested models

importance_R = imp_R[0]

for nr in range(1,len(imp_R)):
    importance_R = pd.concat([importance_R,imp_R[nr]],axis=0) 


#rfpimp.plot_importances(importance_R)

#save the stuff
with open('models/rfcForRedWins.pkl', 'wb') as handle:
    pickle.dump(rfc_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
report_R.to_pickle("resources/rfcReportRedWins.pkl")
importance_R.to_pickle("resources/featuresRedWins.pkl")

#and now do the same thing if blue wins (sanity check)
y = B_Winner_Blue

#use random wood ;) as classifier 
rfc = RandomForestClassifier()
#rfc.fit(X_train, y_train.values.ravel()) ==> instead of normal fitting, we use cross validation to estimate the parameters because test_recall was not so good after fitting

rfc_scores = cross_validate(rfc, X, y.values.ravel(), cv=9, scoring=('accuracy', 'average_precision', 'recall','f1'), return_estimator=True)
model = rfc_scores['estimator']

#make a report on accuracy, precision, recall and f1
report = pd.DataFrame(index=list(rfc_scores.keys())[3:], columns=['Random Forest avg', 'Random Forest std'])
report = pd.DataFrame(index=list(rfc_scores.keys())[3:], columns=['Random Forest avg', 'Random Forest std', 'Chosen model params'])
for key in report.index:
    if(key=="test_recall"):
        #take the model closest do the mean
        checkval = np.abs(np.mean(rfc_scores[key])-rfc_scores[key])
        idx = np.where(np.min(checkval)==checkval)[0][0]
        rfc=model[idx]
        break

for key in report.index:
    report.loc[key] = [np.mean(rfc_scores[key]), np.std(rfc_scores[key]), rfc_scores[key][idx]]

report *= 100
report_B = report.astype(float).round(1)
rfc_B = rfc

#check importance of different features
importance_B = rfpimp.importances(rfc, X_test, y_test)
#rfpimp.plot_importances(importance_R)


#save the stuff
with open('models/rfcForBlueWins.pkl', 'wb') as handle:
    pickle.dump(rfc_B, handle, protocol=pickle.HIGHEST_PROTOCOL)
report_B.to_pickle("resources/rfcReportBlueWins.pkl")
importance_B.to_pickle("resources/featuresBlueWins.pkl")

#report_B has very low recall and f1 ==> not enough data to make a model the other way round and make comparisson
#according to imprtance R: most significant feature is the age of each combatant followed by landed significant strikes
