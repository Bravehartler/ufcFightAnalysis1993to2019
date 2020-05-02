# %%
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
import plotly
import plotly.express as px

import pickle
from matplotlib import cm

def layoutSetting(fig):
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

    fig.update_xaxes(title_font=dict(size=24, family='Courier'), tickfont=dict(family='Courier', size=20))
    fig.update_yaxes(title_font=dict(size=24, family='Courier'), tickfont=dict(family='Courier', size=20))

    fig.update_layout(legend=dict(
        traceorder="normal",
        font=dict(
            family="Courier",
            size=20,
            color="black"
        )
    ))

    return fig


# %%
# get dataframe and do awesome stuff
df = pd.read_pickle("data/clean/clean.pkl")
df_R = pd.DataFrame(df.R_fighter)
df_B = pd.DataFrame(df.B_fighter)

nr_R = 0
nr_B = 0
headers = list(df)
for head in headers:
    if head[:2] == "R_":
        df_R.insert(nr_R, head[2:], df[head])
        nr_R += 1
    elif head[:2] == "B_":
        df_B.insert(nr_B, head[2:], df[head])
        nr_B += 1
    else:
        df_R.insert(nr_R, head, df[head])
        df_B.insert(nr_B, head, df[head])
        nr_R += 1
        nr_B += 1
df_R = df_R.drop(list(df_R)[-1], axis=1)
df_B = df_B.drop(list(df_B)[-1], axis=1)


df_RB = pd.concat([df_R,df_B],axis=0)
df_RB.index = range(0,len(df_RB.index))
df.to_pickle("data/clean/fighterTournamentStats.pkl")
df.to_csv("data/clean/fighterTournamentStats.csv")


# %%

importance_R = pd.read_pickle("resources/featuresRedWins.pkl").reset_index().sort_values(by="Importance", ascending=True)
report_R = pd.read_pickle("resources/rfcReportRedWins.pkl")
# Now rly finally time for some nice plots :,)
my_colors = ['rgba(38, 24, 74, 1)', 'rgba(71, 58, 131, 1)',
    'rgba(122, 120, 168, 1)', 'rgba(164, 163, 204, 1)',
    'rgba(190, 192, 213, 1)']

# Question, What feature determines win/loss of a game
fig = px.bar(importance_R[-5:importance_R.shape[0]], x="Importance", y="Feature", 
    orientation="h", barmode="group",
    labels={'Importance': 'IMPORTANCE', 'Feature': 'FEATURE'}, color_discrete_sequence=my_colors)
fig = layoutSetting(fig)

#plotly.offline.plot(fig)
fig.write_image("resources/winFeatureImportance.png")

# Question, Who is the biggest badass in the ring (most won fights)?
champion = df_RB.groupby(["fighter"]).wins.max().reset_index().sort_values(by="wins", ascending=True)
fig = px.bar(champion[-5:champion.shape[0]], x="wins", y="fighter", 
    orientation="h", barmode="group",
    labels={'fighter': 'FIGHTER', 'wins': 'WINS'}, color_discrete_sequence=my_colors)
fig = layoutSetting(fig)

#plotly.offline.plot(fig)
fig.write_image("resources/mostWinsFighter.png")

# Question, Who won the most championships?
champion = df_RB[df_RB.title_bout==1]
champion = champion.groupby(["fighter"]).wins.max().reset_index().sort_values(by="wins", ascending=True)
fig = px.bar(champion[-5:champion.shape[0]], x="wins", y="fighter", 
    orientation="h", barmode="group",
    labels={'fighter': 'FIGHTER', 'wins': 'WINS'}, color_discrete_sequence=my_colors)

fig = layoutSetting(fig)

#plotly.offline.plot(fig)
fig.write_image("resources/titleFighter.png")

# Question: Which fighter can keep his win streak going and check if it aligns to the most significant feature for wins
#check which fighter has the longest win streak for a certain age
champion_idx = df_RB.groupby(["fighter"]).current_win_streak.idxmax().values
champion = df_RB.loc[champion_idx, ['fighter', 'age', 'current_win_streak']].sort_values(by="current_win_streak", ascending=True)
champion.age = champion.age.astype(str)
fig = px.bar(champion[-5:champion.shape[0]], x="current_win_streak", y="fighter", color="age",
    orientation="h", barmode="group",
    labels={'fighter': 'FIGHTER', 'current_win_streak': 'WIN STREAK', 'age': 'AGE'}, color_discrete_sequence=my_colors)

fig = layoutSetting(fig)

#plotly.offline.plot(fig)
fig.write_image("resources/fighterAgeWinStreak.png")


# Question: Does the found feature for determining wins, align with wins per age?
#check the sum of wins done for different ages
ageDistro = df_RB.groupby(["age"]).wins.sum().sort_index().reset_index()
fig = px.bar(ageDistro, x="age", y="wins",
    orientation="v", barmode="group",
    labels={'wins': 'WINS', 'age': 'AGE'}, color_discrete_sequence=my_colors)

fig = layoutSetting(fig)

#plotly.offline.plot(fig)
fig.write_image("resources/winsToAge.png")
