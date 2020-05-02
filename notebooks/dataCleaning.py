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

#load the previous generated featureGen File and the comparissons
df = pd.read_pickle("./data/clean/featureGen.pkl")
df = pd.read_csv("./data/clean/featureGen.csv")
df_partprufc = pd.read_csv("./data/raw/data.csv")
df_preprufc = pd.read_csv("./data/raw/preprocessed_data.csv")

#drop duplicates and nan
df = df.drop_duplicates()
df = df.dropna()

headers = np.array(list(df))

df = df.drop(headers[0], axis=1)
df.to_pickle("data/clean/clean.pkl")
df.to_csv("data/clean/clean.csv")

drops = ["R_fighter", "B_fighter", "Referee", "date", "location"]

for drop in drops:
    df = df.drop(drop, axis=1)

categorical_columns = ["weight_class"]
for col in categorical_columns:
    df[col] = df[col].astype('category')


# following code is for comparisson reasons atm the STR have a . at the end, preprufc only use winner columen, B_Stance_sideways and weight class open (dropped due to cleaning)
#weight also seems to be present
'''
df = pd.get_dummies(df, drop_first=False)
for i in list(df_preprufc): 
    if(i not in list(df)): print(i)
'''

#normal path
df = pd.get_dummies(df, drop_first=True)
for i in list(df_preprufc): 
    if(i not in list(df)): print(i)

Winner_Red = df["Winner_Red"]
Winner_Draw = df["Winner_Draw"]
df.drop(labels=["Winner_Red", "Winner_Draw"], axis=1, inplace = True)
df.insert(0, 'Winner_Draw', Winner_Draw)
df.insert(0, 'Winner_Red', Winner_Red)

df.to_pickle("data/clean/cleanMLRdy.pkl")
df.to_csv("data/clean/cleanMLRdy.csv")

'''
questions: 
0) How did the guy on Keggle made his pre-processing and constructed his features? => I think I answered this completely in the other python code... man this took long
1) Which one has the overall most wins?
2) which stat/feature is the most representative for winning?
3) Challenge: After finding out this stat, who is the strongest fighter of all time?
4) Similar who is the strongest fighter for each wheight class?
Backups:
- who won title according to date
'''
# %%
