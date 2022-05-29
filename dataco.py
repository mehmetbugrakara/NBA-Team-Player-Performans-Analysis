
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


### PLAYERADVANCED Data ###
playeradvanced2022=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playeradvanced.xlsx")
playeradvanced2021=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2020-2021\playeradvanced.xlsx")
playeradvanced2020=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2019-2020\playeradvanced.xlsx")
playeradvanced2019=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2018-2019\playeradvanced2.xlsx")
playeradvanced2018=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2017-2018\playeradvanced.xlsx")
playeradvanced2017=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2016-2017\playeradvanced.xlsx")

framesAdv=[playeradvanced2021,playeradvanced2020,playeradvanced2019,playeradvanced2018,playeradvanced2017]
TotalPlayerAdvanced=pd.concat(framesAdv,keys=("2021","2020","2019","2018","2017"))
#TotalPlayerAdvanced=TotalPlayerAdvanced[TotalPlayerAdvanced["MP"].values>=150]

### PLAYERPERGame Data ###
playerpergame2022=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playerpergame.xlsx")
playerpergame2021=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2020-2021\playerpergame.xlsx")
playerpergame2020=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2019-2020\playerpergame.xlsx")
playerpergame2019=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2018-2019\playerpergame.xlsx")
playerpergame2018=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2017-2018\playerpergame.xlsx")
playerpergame2017=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2016-2017\playerpergame.xlsx")

framespgame=[playerpergame2021,playerpergame2020,playerpergame2019,playerpergame2018,playerpergame2017]
TotalPlayerpergame=pd.concat(framespgame,keys=("2021","2020","2019","2018","2017"))

TotalPlayerpergame=TotalPlayerpergame.rename({'MP':'MPG','Pos':'Pos1'},axis=1)

playerpergame2022=playerpergame2022.rename({'MP':'MPG','Pos':'Pos1'},axis=1)

### PLAYERPER100 Data ###
playerper1002022=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playerper100.xlsx")
playerper1002021=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2020-2021\playerper100.xlsx")
playerper1002020=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2019-2020\playerper100.xlsx")
playerper1002019=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2018-2019\playerper100.xlsx")
playerper1002018=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2017-2018\playerper100.xlsx")
playerper1002017=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2016-2017\playerper100.xlsx")

framesp100=[playerper1002021,playerper1002020,playerper1002019,playerper1002018,playerper1002017]
TotalPlayerper100=pd.concat(framesp100,keys=("2021","2020","2019","2018","2017"))
#TotalPlayerper100=pd.concat(framesp100)

TotalPlayerper100Pos=TotalPlayerper100.iloc[:,29:31]
TotalPlayerper100.info()

playerper1002022=playerper1002022.iloc[:,29:31]

### TAKIM VERİLERİ ###

### Team advanced data ###

teamadvanced2022=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2021-2022\teamadvanced.xlsx")
teamadvanced2021=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2020-2021\teamadvanced.xlsx")
teamadvanced2020=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2019-2020\teamadvanced.xlsx")
teamadvanced2019=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2018-2019\teamadvanced.xlsx")
teamadvanced2018=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2017-2018\teamadvanced.xlsx")
teamadvanced2017=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2016-2017\teamadvanced.xlsx")

framesteamAdv=[teamadvanced2021,teamadvanced2020,teamadvanced2019,teamadvanced2018,teamadvanced2017]
TotalteamAdvanced=pd.concat(framesteamAdv)


### Team pergame off data ###

teampergameoff22=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2021-2022\teampergameoff.xlsx")

teampergameoff21=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2020-2021\teampergameoff.xlsx")
teampergameoff20=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2019-2020\teampergameoff.xlsx")
teampergameoff19=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2018-2019\teampergameoff.xlsx")
teampergameoff18=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2017-2018\teampergameoff.xlsx")
teampergameoff17=pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2016-2017\teampergameoff.xlsx")

framesteampergame=[teampergameoff21,teampergameoff20,teampergameoff19,teampergameoff18,teampergameoff17]
Totalteampergame=pd.concat(framesteampergame)

def sortvalues(teamvalue):
    teamvalue=teamvalue.sort_values(by=['Team'])
    teamvalue=teamvalue.reset_index()
    teamvalue=teamvalue.drop(columns=(['index','Rk']))
    
    
    return teamvalue
    


TotalteamAdvanced=sortvalues(TotalteamAdvanced)
Totalteampergame=sortvalues(Totalteampergame)
teampergameoff22=sortvalues(teampergameoff22)
teamadvanced2022=sortvalues(teamadvanced2022)


# TotalteamAdvanced=TotalteamAdvanced.drop([30]) ### droping league average
# Totalteampergame=Totalteampergame.drop([30])
Totalteampergame=Totalteampergame[Totalteampergame.Team !='League Average']
TotalteamAdvanced=TotalteamAdvanced[TotalteamAdvanced.Team !='League Average']
frames=[TotalPlayerAdvanced,TotalPlayerpergame,TotalPlayerper100Pos]     
TotalPlayerData=pd.concat(frames,axis=1)

TotalPlayerData=TotalPlayerData.loc[:,~TotalPlayerData.T.duplicated()]  



corrMatrix = TotalPlayerData.corr()
corrMatrix1 = TotalteamAdvanced.corr()
# sns.heatmap(corrMatrix, annot=True)

# TotalPlayerData.to_csv('totalplayerdata.csv')

def birlestir(pergame,advanced):
    
    locper=pergame.loc[:,['AST','PTS','TRB','TOV']]
    locadv=advanced.loc[:,['W','L','ORtg','DRtg','NRtg']]
    
    birlesik=pd.concat([locper,locadv],axis=1)
    
    return birlesik


trainteam=birlestir(Totalteampergame,TotalteamAdvanced)
testteam=birlestir(teampergameoff22,teamadvanced2022)

def plotvalue(xdegeri,ydegeri,data,hue):
    sns.lmplot(x=xdegeri, y=ydegeri, data=data,hue=hue)
    plt.title(xdegeri+' to '+ydegeri)
    
    
# plotvalue('W', 'NRtg', teamadvanced2022, 'Team')
    

sns.lmplot(x='W', y='DRtg', data=teamadvanced2022)
plt.title('Win Based on Defensive Rating')


sns.lmplot(x='W', y='ORtg', data=teamadvanced2022)
plt.title('Win Based on Offensive Rating')
sns.lmplot(x='W', y='NRtg', data=teamadvanced2022)
plt.title('Win Based on Net Rating')
# trainteam.to_csv('trainteam.csv')
# testteam.to_csv('testteam.csv')





