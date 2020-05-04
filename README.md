# ufcFightAnalysis1993to2019

General folder structure:

    ├── data                #downloaded and cleaned data    
    │   ├── clean           #cleaned data
    │   └── raw             #raw UFC fight data
    ├── models              #built models
    ├── notebooks           #used python files
    ├── reports             #reports on the analyzed UFC data
    └── resources           #used resources in the report
        ├── graphs          #graphs generated
        ├── modelOutcome    #reports and importance feature on the used models
        └── Pictures        #genreal pictures used in the report
        
       
 Used dataset for analysis:
 https://www.kaggle.com/rajeevw/ufcdata#data.csv
 
 This is a small Data Science project to analyze data from UFC fights between 1993 to 2019 in order to make predictions on the outcome of a fight.
 Therefore, from the original raw datasets features were built, which give fight statistics and stats of the fighters themselves.
 After data cleaning, a random forrest classifier was used to determine the most important features responsible for a win of the fighter in question (in this case it will be referred to as the red fighter)
 Determining factors were further analyzed to assign a quantitative value to the favoured fighter.
 The outcome is described in further detail in the file UFCDataAnalysisReport.pdf inside of the reports folder.
