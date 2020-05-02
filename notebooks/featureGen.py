# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
import plotly.express as px


# %%
df_partprufc = pd.read_csv("./data/raw/data.csv")
# Description: "This is the partially processed file. All feature engineering has been included and every row is a compilation of info about each fighter up until that fight. The data has not been one hot encoded or processed for missing data. You can use this file to do your own processing and further feature engineering."

df_preprufc = pd.read_csv("./data/raw/preprocessed_data.csv")
# Description: "This is the preprocessed file where each row contains information about each fighter collected up until this particular fight and the descriptive fight details. In this file, feature engineering, missing data treatment and one hot encoding are already done. The raw files have been merged to form this dataset."

df_rawufc = pd.read_csv("./data/raw/raw_fighter_details.csv")
#Description: "This is the raw fighter details file which contains details about each fighter scraped from the ufcstats website"

df_ufc = pd.read_csv("./data/raw/raw_total_fight_data.csv",delimiter=";")
#Description: " Raw fight data scraped from ufcstats events page. This contains scraped information of every fight of every event in the ufc."

# %%
# Since you should never trust data that you did not manipulate yourself (except maybe colleagues because you know their adresses :) ), we start out with the df_ufc in order to check what the author did to his data sets ==> First question: How did he received his dataframe features?
# Focus on df_ufc and let's start out with data cleaning and processing (also compare it with partprufc and preprufc):

#first check headers:
headers = np.array(list(df_ufc))
headers_part = np.array(list(df_partprufc))


df_ufc.date = pd.to_datetime(df_ufc.date)
df_rawufc.DOB = pd.to_datetime(df_rawufc.DOB)
df_ufc = df_ufc.iloc[::-1] #seems like data collector already ordered data => date is in descending order atm so we need to reverse for later use

#create new improved dataframe obtaining the features
df = pd.DataFrame(df_ufc.R_fighter)
df.insert(1, 'B_fighter', df_ufc.B_fighter)
df.insert(2, 'Referee', df_ufc.Referee)
df.insert(3, 'date', df_ufc.date)
df.insert(4, 'location', df_ufc.location)
df.insert(5, 'Winner', df_ufc.Winner)




title_bout = df_ufc.Fight_type.copy()
weight_class = df_ufc.Fight_type.copy()

WeightClasses = [
    "Catch Weight",
    "Open Weight",
    "Light Heavyweight",
    "Flyweight",
    "Strawweight",
    "Featherweight",
    "Lightweight",
    "Middleweight",
    "Bantamweight",
    "Welterweight",
    "Heavyweight",
]

no_of_rounds = df_ufc.Format.copy()

B_current_lose_streak = df_ufc.Winner.copy()
B_current_win_streak = df_ufc.Winner.copy()
B_draw = df_ufc.Winner.copy()
B_wins = df_ufc.Winner.copy()
B_losses = df_ufc.Winner.copy()
B_longest_win_streak = df_ufc.Winner.copy()
B_longest_win_streak.loc[:] = 0.0
B_total_rounds_fought = B_longest_win_streak.copy()
B_total_time_fought_seconds = B_longest_win_streak.copy()
B_total_title_bouts = B_longest_win_streak.copy()

B_Stance = df_ufc.Winner.copy()
B_Height_cms = df_ufc.Winner.copy()
B_Reach_cms = df_ufc.Winner.copy()
B_Weight_lbs = df_ufc.Winner.copy()
B_age = df_ufc.Winner.copy()


R_current_lose_streak = df_ufc.Winner.copy()
R_current_win_streak = df_ufc.Winner.copy()
R_draw = df_ufc.Winner.copy()
R_wins = df_ufc.Winner.copy()
R_losses = df_ufc.Winner.copy()
R_longest_win_streak = df_ufc.Winner.copy()
R_longest_win_streak.loc[:] = 0.0
R_total_rounds_fought = R_longest_win_streak.copy()
R_total_time_fought_seconds = R_longest_win_streak.copy()
R_total_title_bouts = R_longest_win_streak.copy()

R_Stance = df_ufc.Winner.copy()
R_Height_cms = df_ufc.Winner.copy()
R_Reach_cms = df_ufc.Winner.copy()
R_Weight_lbs = df_ufc.Winner.copy()
R_age = df_ufc.Winner.copy()

#for lose win draw count
#Here it is important to go from the first tracked fight to the newest one in order to count different features
fighters = np.unique(np.concatenate((df_ufc.R_fighter,df_ufc.B_fighter))) #less fighters than the df_rawufc => next for will 
currentWins = np.zeros(np.size(fighters))
currentLosses = np.zeros(np.size(fighters))
totalWins = np.zeros(np.size(fighters))
totalLosses = np.zeros(np.size(fighters))
totalDraws = np.zeros(np.size(fighters))
longestWins = np.zeros(np.size(fighters))

totalRounds = np.zeros(np.size(fighters))
totalTime = np.zeros(np.size(fighters))
totalTitleBout = np.zeros(np.size(fighters))

matches_count = np.zeros(np.size(fighters))






#atk counter ==> so we make the assumption that the fighter did not make any attacks whatsoever
atk_str = ['BODY', 'CLINCH', 'DISTANCE', 'GROUND', 'HEAD', 'LEG', 'SIG_STR.', 'TOTAL_STR.', 'TD']
atk_substr = ['_att', '_landed']
atk_type = {}
atk_type_opp = {}
df_B_atk_type = {}
df_R_atk_type = {}
df_B_atk_type_opp = {}
df_R_atk_type_opp = {}
for atk in atk_str:
    for suFix in atk_substr:
        atk_type[atk+suFix] = np.zeros(np.size(fighters))
        atk_type_opp[atk+suFix] = np.zeros(np.size(fighters))
        df_B_atk_type[atk+suFix] = df_ufc.B_BODY.copy()
        df_R_atk_type[atk+suFix] = df_ufc.R_BODY.copy()
        df_B_atk_type_opp[atk+suFix] = df_ufc.B_BODY.copy()
        df_R_atk_type_opp[atk+suFix] = df_ufc.R_BODY.copy()


attacksSpecial = ["KD", "PASS", "REV", "SIG_STR_pct", "SUB_ATT", "TD_pct"]

special_type = {}
special_type_opp = {}
df_B_special_type = {}
df_R_special_type = {}
df_B_special_type_opp = {}
df_R_special_type_opp = {}
for atk in attacksSpecial:
    special_type[atk] = np.zeros(np.size(fighters))
    special_type_opp[atk] = np.zeros(np.size(fighters))
    df_B_special_type[atk] = df_ufc.B_BODY.copy()
    df_R_special_type[atk] = df_ufc.R_BODY.copy()
    df_B_special_type_opp[atk] = df_ufc.B_BODY.copy()
    df_R_special_type_opp[atk] = df_ufc.R_BODY.copy()


winTypesStr = df_ufc.win_by.unique()
for nr in range(0,len(winTypesStr)):
    winTypesStr[nr] = winTypesStr[nr].replace(' - ','_')
    winTypesStr[nr] = winTypesStr[nr].replace("'s ",'_')

winTypes = {}
df_B_winTypes = {}
df_R_winTypes = {}
for winStr in winTypesStr:
    winTypes[winStr] = np.zeros(np.size(fighters))
    df_B_winTypes[winStr] = df_ufc.B_BODY.copy()
    df_R_winTypes[winStr] = df_ufc.R_BODY.copy()

InchToCm = 2.54
FeetinCm = 30.48

for nr in df.index:
    
    #fighter properties
    check = df_rawufc.fighter_name == df.B_fighter[nr]
    df_part = df_rawufc[check]
    if(len(df_part.Stance)==0):
        B_Stance[nr] = np.nan
    else:
        B_Stance[nr] = df_part.Stance.values[0]
    if(df_part.Height.isna().any()):
        B_Height_cms[nr] = np.nan
    else:
        if(len(df_part.Height)==0):
            B_Height_cms[nr] = np.nan
        else:
            Height = df_part.Height.values[0].replace('"', '').replace("\'", "").split(" ")
            B_Height_cms[nr] = float(Height[0])*FeetinCm + float(Height[1])*InchToCm
    
    if(df_part.Reach.isna().any()):
        B_Reach_cms[nr] = np.nan
    else:
        if(len(df_part.Reach)==0):
            B_Reach_cms[nr] = np.nan
        else:
            Reach = df_part.Reach.values[0].replace('"', '').replace("\'", "").split(" ")
            B_Reach_cms[nr] = float(Reach[0])*InchToCm
    if(len(df_part.Weight)==0):
        B_Weight_lbs[nr] = np.nan
    else:
        if(df_part.Weight.isna().any()):
            B_Weight_lbs[nr] = np.nan
        else:
            B_Weight_lbs[nr] = float(df_part.Weight.values[0].replace(" lbs.",""))
    if(len(df_part.DOB)==0):
        R_age[nr] = np.nan
    else:
        if (df_ufc.date-df_part.DOB.values[0]).isna()[nr]:
            B_age[nr] = np.nan
        else:
            B_age[nr] = int((df_ufc.date-df_part.DOB.values[0])[nr]/np.timedelta64(1,'Y'))


    check = df_rawufc.fighter_name == df.R_fighter[nr]
    df_part = df_rawufc[check]
    if(len(df_part.Stance)==0):
        R_Stance[nr] = np.nan
    else:
        R_Stance[nr] = df_part.Stance.values[0]
    if(df_part.Height.isna().any()):
        R_Height_cms[nr] = np.nan
    else:
        if(len(df_part.Height)==0):
            R_Height_cms[nr] = np.nan
        else:
            Height = df_part.Height.values[0].replace('"', '').replace("\'", "").split(" ")
            R_Height_cms[nr] = float(Height[0])*FeetinCm + float(Height[1])*InchToCm
    
    if(df_part.Reach.isna().any()):
        R_Reach_cms[nr] = df_part.Reach.values[0]
    else:
        if(len(df_part.Reach)==0):
            R_Reach_cms[nr] = np.nan
        else:
            Reach = df_part.Reach.values[0].replace('"', '').replace("\'", "").split(" ")
            R_Reach_cms[nr] = float(Reach[0])*InchToCm
    if(len(df_part.Weight)==0):
        R_Weight_lbs[nr] = np.nan
    else:
        if(df_part.Weight.isna().any()):
            R_Weight_lbs[nr] = np.nan
        else:
            R_Weight_lbs[nr] = float(df_part.Weight.values[0].replace(" lbs.",""))
    if(len(df_part.DOB)==0):
        R_age[nr] = np.nan
    else:
        if (df_ufc.date-df_part.DOB.values[0]).isna()[nr]:
            R_age[nr] = np.nan
        else:
            R_age[nr] = int((df_ufc.date-df_part.DOB.values[0])[nr]/np.timedelta64(1,'Y'))

    
    # WIN AND LOSS STUFF
    #-----------------------------------------------------------------------------------------------
    #Input wins losses draws for each fighter
    #START with blue
    check = [fighters == df.B_fighter[nr]]
    B_current_lose_streak[nr] = currentLosses[check][0]
    B_current_win_streak[nr] = currentWins[check][0]
    B_draw[nr] = totalDraws[check][0]
    B_wins[nr] = totalWins[check][0]
    B_losses[nr] = totalLosses[check][0]
    B_total_rounds_fought[nr] = totalRounds[check][0]
    B_total_time_fought_seconds[nr] = totalTime[check][0]
    B_total_title_bouts[nr] = totalTitleBout[check][0]
    for winStr in winTypesStr:
        df_B_winTypes[winStr][nr] = winTypes[winStr][check][0]
    B_longest_win_streak[nr] = max(B_longest_win_streak[nr],longestWins[check][0]) #TODO
    
    #THEN Red
    check = [fighters == df.R_fighter[nr]]
    R_current_lose_streak[nr] = currentLosses[check][0]
    R_current_win_streak[nr] = currentWins[check][0]
    R_draw[nr] = totalDraws[check][0]
    R_wins[nr] = totalWins[check][0]
    R_losses[nr] = totalLosses[check][0]
    R_total_rounds_fought[nr] = totalRounds[check][0]
    R_total_time_fought_seconds[nr] = totalTime[check][0]
    R_total_title_bouts[nr] = totalTitleBout[check][0]
    for winStr in winTypesStr:
        df_R_winTypes[winStr][nr] = winTypes[winStr][check][0]
    R_longest_win_streak[nr] = max(R_longest_win_streak[nr],longestWins[check][0]) #TODO

    #Change winner to red or blue
    #calculate wins losses and draws
    if (df.Winner[nr] == df.R_fighter[nr]):
        currentWins[fighters == df.B_fighter[nr]] = 0
        currentLosses[fighters == df.B_fighter[nr]] += 1
        totalLosses[fighters == df.B_fighter[nr]] += 1

        currentWins[fighters == df.Winner[nr]] += 1
        currentLosses[fighters == df.Winner[nr]] = 0
        totalWins[fighters == df.Winner[nr]] += 1

        df.Winner[nr] = "Red"
    elif (df.Winner[nr] == df.B_fighter[nr]):
        currentWins[fighters == df.R_fighter[nr]] = 0
        currentLosses[fighters == df.R_fighter[nr]] += 1
        totalLosses[fighters == df.R_fighter[nr]] += 1

        currentWins[fighters == df.Winner[nr]] += 1
        currentLosses[fighters == df.Winner[nr]] = 0
        totalWins[fighters == df.Winner[nr]] += 1

        df.Winner[nr] = "Blue"
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # WTF: This part is the most weirdes I've ever seen, the guy who made this preprocessed the df_partprufc such
        # that this does not account as draw since it is nan
        # So far so good BUT he acount this as reset for the current wins and it counts as loss
        # I will keep it as it is in my code for comparisson reasons and documentation of this FAIL
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #totalDraws[fighters == df.R_fighter[nr]] += 1
        #totalDraws[fighters == df.B_fighter[nr]] += 1
        #draw sets winstreak back
        currentWins[fighters == df.B_fighter[nr]] = 0
        currentWins[fighters == df.R_fighter[nr]] = 0
        
        #count as loss (oh man)
        currentLosses[fighters == df.B_fighter[nr]] += 1
        currentLosses[fighters == df.R_fighter[nr]] += 1
        #count as overall loss
        totalLosses[fighters == df.B_fighter[nr]] += 1
        totalLosses[fighters == df.R_fighter[nr]] += 1

        df.Winner[nr] = "Draw"
    
    # TODO
    # So i tried a lot for the longest win streak and cant get my head wrapped arround it. The most strange thing
    # is that when you group the dataframes by B_fighter and compare the maximum of the current win streaks from my 
    # generated dataset, then there are differences with his dataset df_partprufc but if you compare the current win
    # streaks of both sets with each other they perfectly match
    # anyways i will keep it as it is now, as the maximum of the current win streaks to the recorded win streak
    if (df.Winner[nr] == df.R_fighter[nr]):
        longestWins[fighters == df.B_fighter[nr]] = 0
        longestWins[fighters == df.Winner[nr]] += 1
    elif (df.Winner[nr] == df.B_fighter[nr]):
        longestWins[fighters == df.R_fighter[nr]] = 0
        longestWins[fighters == df.Winner[nr]] += 1
    else:
        longestWins[fighters == df.B_fighter[nr]] = 0
        longestWins[fighters == df.R_fighter[nr]] = 0
    

    
    #-----------------------------------------------------------------------------------------------------
    #winBys
    check = [fighters == df_ufc.Winner[nr]]
    checkStr = df_ufc.win_by[nr]
    checkStr = checkStr.replace(' - ','_')
    checkStr = checkStr.replace("'s ",'_')
    winTypes[checkStr][check] += 1

    #-----------------------------------------------------------------------------------------------------
    #title_bout
    if "Title Bout" in df_ufc.Fight_type[nr]:
        title_bout[nr] = 1

        check = [fighters == df.B_fighter[nr]]
        totalTitleBout[check] += 1
        check = [fighters == df.R_fighter[nr]]
        totalTitleBout[check] += 1
    elif "Bout" in df_ufc.Fight_type[nr]:
        title_bout[nr] = 0
    else:
        title_bout[nr] = np.nan
    
    #-----------------------------------------------------------------------------------------------------
    #weight_class
    preFix = ""
    if "Women's" in weight_class[nr]:
        preFix = "Women's "
        
    for weight in WeightClasses:
        if weight in df_ufc.Fight_type[nr]:
            weight_class[nr] = preFix + weight
            break
        else:
            weight_class[nr] = "Catch Weight"
    
    #-----------------------------------------------------------------------------------------------------
    #no of rounds
    if ('No Time Limit' in no_of_rounds[nr]):
        no_of_rounds[nr] = "1"
    no_of_rounds[nr] = float(no_of_rounds[nr][0])

    check = [fighters == df.B_fighter[nr]]
    totalRounds[check] += float(df_ufc['last_round'][nr])
    time = df_ufc['last_round_time'][nr].split(":")
    time = float(time[0])*60 + float(time[1])
    totalTime[check] += time

    check = [fighters == df.R_fighter[nr]]
    totalRounds[check] += float(df_ufc['last_round'][nr])
    time = df_ufc['last_round_time'][nr].split(":")
    time = float(time[0])*60 + float(time[1])
    totalTime[check] += time

    
    #-----------------------------------------------------------------------------------------------------
    #Body atks: (calc before fight)
    for atk_stringPart in atk_str:
        #BLUE FIGHTER
        check = [fighters == df.B_fighter[nr]]
        df_B_atk_type[atk_stringPart+"_landed"][nr] = atk_type[atk_stringPart+"_landed"][check][0]/matches_count[check][0]
        df_B_atk_type[atk_stringPart+"_att"][nr] = atk_type[atk_stringPart+"_att"][check][0]/matches_count[check][0]
        
        atk = df_ufc["B_"+atk_stringPart][nr].split("of")
        atk_type[atk_stringPart+"_landed"][check] += float(atk[0])
        atk_type[atk_stringPart+"_att"][check] += float(atk[1])

        #BLUE FIGHTER opponent stats
        df_B_atk_type_opp[atk_stringPart+"_landed"][nr] = atk_type_opp[atk_stringPart+"_landed"][check][0]/matches_count[check][0]
        df_B_atk_type_opp[atk_stringPart+"_att"][nr] = atk_type_opp[atk_stringPart+"_att"][check][0]/matches_count[check][0]
        

        atk = df_ufc["R_"+atk_stringPart][nr].split("of")
        atk_type_opp[atk_stringPart+"_landed"][check] += float(atk[0])
        atk_type_opp[atk_stringPart+"_att"][check] += float(atk[1])

        #RED FIGHTER
        check = [fighters == df.R_fighter[nr]]
        df_R_atk_type[atk_stringPart+"_landed"][nr] = atk_type[atk_stringPart+"_landed"][check][0]/matches_count[check][0]
        df_R_atk_type[atk_stringPart+"_att"][nr] = atk_type[atk_stringPart+"_att"][check][0]/matches_count[check][0]
        
        atk = df_ufc["R_"+atk_stringPart][nr].split("of")
        atk_type[atk_stringPart+"_landed"][check] += float(atk[0])
        atk_type[atk_stringPart+"_att"][check] += float(atk[1])

        #RED FIGHTER opponent stats
        check = [fighters == df.R_fighter[nr]]
        df_R_atk_type_opp[atk_stringPart+"_landed"][nr] = atk_type_opp[atk_stringPart+"_landed"][check][0]/matches_count[check][0]
        df_R_atk_type_opp[atk_stringPart+"_att"][nr] = atk_type_opp[atk_stringPart+"_att"][check][0]/matches_count[check][0]
        
        atk = df_ufc["B_"+atk_stringPart][nr].split("of")
        atk_type_opp[atk_stringPart+"_landed"][check] += float(atk[0])
        atk_type_opp[atk_stringPart+"_att"][check] += float(atk[1])
    
    #-----------------------------------------------------------------------------------------------------
    #SPECIAL ATTACKS
    for atk_stringPart in attacksSpecial:
        #BLUE FIGHTER
        check = [fighters == df.B_fighter[nr]]
        df_B_special_type[atk_stringPart][nr] = special_type[atk_stringPart][check][0]/matches_count[check][0]
        df_B_special_type[atk_stringPart][nr] = special_type[atk_stringPart][check][0]/matches_count[check][0]
        
        atk = df_ufc["B_"+atk_stringPart][nr]
        if(isinstance(atk, str)):
            atk = float(atk.replace("%",""))/100
        else:
            atk = float(atk)
        special_type[atk_stringPart][check] += atk

        #BLUE FIGHTER opponent stats
        df_B_special_type_opp[atk_stringPart][nr] = special_type_opp[atk_stringPart][check][0]/matches_count[check][0]
        df_B_special_type_opp[atk_stringPart][nr] = special_type_opp[atk_stringPart][check][0]/matches_count[check][0]
        

        atk = df_ufc["R_"+atk_stringPart][nr]
        if(isinstance(atk, str)):
            atk = float(atk.replace("%",""))/100
        else:
            atk = float(atk)
        special_type_opp[atk_stringPart][check] += atk

        #RED FIGHTER
        check = [fighters == df.R_fighter[nr]]
        df_R_special_type[atk_stringPart][nr] = special_type[atk_stringPart][check][0]/matches_count[check][0]
        df_R_special_type[atk_stringPart][nr] = special_type[atk_stringPart][check][0]/matches_count[check][0]
        
        atk = df_ufc["R_"+atk_stringPart][nr]
        if(isinstance(atk, str)):
            atk = float(atk.replace("%",""))/100
        else:
            atk = float(atk)
        special_type[atk_stringPart][check] += atk

        #RED FIGHTER opponent stats
        check = [fighters == df.R_fighter[nr]]
        df_R_special_type_opp[atk_stringPart][nr] = special_type_opp[atk_stringPart][check][0]/matches_count[check][0]
        df_R_special_type_opp[atk_stringPart][nr] = special_type_opp[atk_stringPart][check][0]/matches_count[check][0]
        
        atk = df_ufc["B_"+atk_stringPart][nr]
        if(isinstance(atk, str)):
            atk = float(atk.replace("%",""))/100
        else:
            atk = float(atk)
        special_type_opp[atk_stringPart][check] += atk
    
    
    #-----------------------------------------------------------------------------------------------------
    # counting matches for the fighter (final)
    check = [fighters == df.B_fighter[nr]]
    matches_count[check] += 1
    check = [fighters == df.R_fighter[nr]]
    matches_count[check] += 1
    #-----------------------------------------------------------------------------------------------------
    
#current win/lose streakbe faster
#B_current_win_streak
#B_current_lose_streak

df_wins = df.copy()
#df_wins.groupBy[]
    
df.insert(6, 'title_bout', title_bout)
df.insert(7, 'weight_class', weight_class)
df.insert(8, 'no_of_rounds', no_of_rounds)
df.insert(9, 'B_current_lose_streak', B_current_lose_streak)
df.insert(10, 'B_current_win_streak', B_current_win_streak)
df.insert(11, 'B_draw', B_draw)

nr = 11

df.insert(nr, 'R_current_lose_streak', B_current_lose_streak)
nr += 1
df.insert(nr, 'R_current_win_streak', B_current_win_streak)
nr += 1
df.insert(nr, 'R_draw', B_draw)

for atk in atk_str:
    for suFix in atk_substr:
        nr += 1
        df.insert(nr, "B_avg_" + atk + suFix, df_B_atk_type[atk+suFix])

for atk in atk_str:
    for suFix in atk_substr:
        nr += 1
        df.insert(nr, "B_avg_opp_" + atk + suFix, df_B_atk_type_opp[atk+suFix])


for atk in attacksSpecial:
    nr += 1
    df.insert(nr, "B_avg_" + atk, df_B_special_type[atk])

for atk in attacksSpecial:
    nr += 1
    df.insert(nr, "B_avg_opp_" + atk, df_B_special_type_opp[atk])


for atk in atk_str:
    for suFix in atk_substr:
        nr += 1
        df.insert(nr, "R_avg_" + atk + suFix, df_R_atk_type[atk+suFix])

for atk in atk_str:
    for suFix in atk_substr:
        nr += 1
        df.insert(nr, "R_avg_opp_" + atk + suFix, df_R_atk_type_opp[atk+suFix])

for atk in attacksSpecial:
    nr += 1
    df.insert(nr, "R_avg_" + atk, df_R_special_type[atk])

for atk in attacksSpecial:
    nr += 1
    df.insert(nr, "R_avg_opp_" + atk, df_R_special_type_opp[atk])

nr+=1
df.insert(nr, 'B_losses', B_losses)
nr+=1
df.insert(nr, 'R_losses', R_losses)


nr+=1
df.insert(nr, 'B_longest_win_streak', B_longest_win_streak)
nr+=1
df.insert(nr, 'R_longest_win_streak', R_longest_win_streak)

nr+=1
df.insert(nr, 'B_total_rounds_fought', B_total_rounds_fought)
nr+=1
df.insert(nr, 'R_total_rounds_fought', R_total_rounds_fought)

nr+=1
df.insert(nr, 'B_total_time_fought(seconds)', B_total_time_fought_seconds)
nr+=1
df.insert(nr, 'R_total_time_fought(seconds)', R_total_time_fought_seconds)

nr+=1
df.insert(nr, 'B_total_title_bouts', B_total_title_bouts)
nr+=1
df.insert(nr, 'R_total_title_bouts', R_total_title_bouts)

nr+=1
df.insert(nr, 'B_wins', B_wins)
nr+=1
df.insert(nr, 'R_wins', R_wins)


winTypesStr = winTypesStr[winTypesStr!='Other']
winTypesStr = winTypesStr[winTypesStr!='Could Not Continue']
winTypesStr = winTypesStr[winTypesStr!='DQ']
winTypesStr = winTypesStr[winTypesStr!='Overturned']
for winStr in winTypesStr:
    nr+=1
    df.insert(nr, "B_win_by_"+winStr, df_B_winTypes[winStr])
    nr+=1
    df.insert(nr, "R_win_by_"+winStr, df_R_winTypes[winStr])

nr+=1
df.insert(nr, 'B_Stance', B_Stance)
nr+=1
df.insert(nr, 'B_Height_cms', B_Height_cms)
nr+=1
df.insert(nr, 'B_Reach_cms', B_Reach_cms)
nr+=1
df.insert(nr, 'B_Weight_lbs', B_Weight_lbs)
nr+=1
df.insert(nr, 'B_age', B_age)

nr+=1
df.insert(nr, 'R_Stance', R_Stance)
nr+=1
df.insert(nr, 'R_Height_cms', R_Height_cms)
nr+=1
df.insert(nr, 'R_Reach_cms', R_Reach_cms)
nr+=1
df.insert(nr, 'R_Weight_lbs', R_Weight_lbs)
nr+=1
df.insert(nr, 'R_age', R_age)

df = df.sort_index()

df.to_pickle("data/clean/featureGen.pkl")
df.to_csv("data/clean/featureGen.csv")

