import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
estimatedteam=pd.read_csv('estimatedteam.csv')

estimatedteam=estimatedteam.drop(columns=['Unnamed: 0'])

x=pd.DataFrame()

y=['W','L']

def sumer(a,b):
    return a+b

estimatedteam['toplam'] = estimatedteam.apply(lambda x: sumer(x['DRtg'], x['ORtg']), axis=1)

Teams=['ATL','BOS','BRK','CHI','CHO','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']

team=pd.DataFrame(data=Teams,columns=['Team'])
for i in range(len(estimatedteam)):
    
    x.loc[i,0]=(82*estimatedteam.iloc[i,0]/estimatedteam.iloc[i,2])
    x.loc[i,1]=82-x.loc[i,0]
    

x.rename(columns = {'0':'W', '1':'L'}, inplace = True)

FinalResult=pd.concat([team,x],axis=1)

sns.scatterplot(x=0, y=1, data=FinalResult,hue='Team')

 

