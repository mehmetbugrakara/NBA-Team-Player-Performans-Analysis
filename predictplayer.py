
import pandas as pd
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import seaborn as sns

Predictedplayers=pd.read_csv('predictedplayers.csv')
Predictedplayers1=pd.read_csv('predictedplayer1.csv')

trainteam=pd.read_csv('trainteam.csv')
testteam=pd.read_csv('testteam.csv')

totalplayerdata=pd.read_csv('totalplayerdata.csv')


houndredpos=totalplayerdata.groupby(["Player"],as_index=False)["ORtg","DRtg"].mean()


mydata=[]
maxdata=[]
DRTgdata=[]
oyuncu=[]
newdata=[]
newest=[]

sumdata=pd.DataFrame(columns=["AST","PTS","TRB","TOV"])
meandata=pd.DataFrame(columns=["ORtg",'DRtg'])
ortalama=pd.DataFrame()
h=0
a=0
b=0
c=0
count=0


Teams=['ATL','BOS','BRK','CHI','CHO','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
Columns=['AST','PTS','TRB','ORtg','DRtg','TOV']


# & (Predictedplayers["MP"] >= 950)
# (Predictedplayers["G"]>=50)

for i in range(len(Teams)):
    
    mydata.append(Predictedplayers1.loc[(Predictedplayers1['Tm']==Teams[i])])   
    maxdata.append(Predictedplayers1.loc[(Predictedplayers1['Tm']==Teams[i]) & (Predictedplayers1["MPG"] >= 28)])
   
    players=mydata[i]['Player']
    players=players.reset_index(drop='index')
    # players=players.values.tolist()
    
    
    for j in range(len(players)):
        
        DRTgdata.append(houndredpos.loc[(houndredpos['Player']==players[j])])
        h=h+1
        
        
        if(houndredpos.loc[(houndredpos['Player']==players[j])].size==0):
            count=count+1
            
        
       
    
    b=b+j+1
    newdata.append(pd.concat(DRTgdata[a:b]))
    
    newdata[i]
    # newest.append(DRTgdata[a:b]).mean()
    a=a+j+1
    

    
    
    for j in range(len(Columns)):
        colum=Columns[j]
        if (Columns[j]=='ORtg' or Columns[j]=='DRtg'):
            
            meandata.loc[i,colum]=(newdata[i][colum].head(8).mean())
        else:
            mydata[i]=mydata[i].sort_values(by='MP',ascending=False)
            sumdata.loc[i,colum]=(mydata[i][colum].head(9).sum())
            
      

def playersname(data,i):
    players=data[i]['Player']
    players=pd.DataFrame(players)
    players=players.reset_index(drop='index')

    players=players.loc[i,:]
    
    
    
    return players
    

def plotvalue(xdegeri,ydegeri,data,hue):
    sns.scatterplot(x=xdegeri, y=ydegeri, data=data,hue=hue)
    plt.title(xdegeri+' to '+ydegeri)
    

plotvalue('G', 'ORTg', Predictedplayers,'Tm')

corrMatrix = Predictedplayers.corr()
sns.heatmap(corrMatrix, annot=True)




# plt.hist(dene['DRTg'],dene['MP'])

# sns.histplot(x="MP", y="ORTg", data=dene,hue=('Pos'))
# sns.lineplot(x="MP", y="ORTg", data=dene1,hue=('Player'))
# plt.xlabel('player')
# plt.ylabel('pts')


    
Team=pd.DataFrame(Teams,columns=['Team'])
skill=pd.concat([sumdata,meandata],axis=1)


Totaldata=pd.concat([Team,skill],axis=1)
def diff(a, b):
    return b - a

Totaldata['NRtg'] = Totaldata.apply(lambda x: diff(x['DRtg'], x['ORtg']), axis=1)

tester=Totaldata.loc[:,['AST','PTS','TRB','TOV','ORtg','DRtg','NRtg']]


bagimli=trainteam.loc[:,['W','L']]
bagimsiz=trainteam.loc[:,['AST','PTS','TRB','TOV','ORtg','DRtg','NRtg']]

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(bagimsiz,bagimli,test_size=0.3, random_state=0)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
X_train = scaler.fit_transform(bagimsiz)
X_test1 = scaler.fit_transform(tester)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


estimater=Sequential()

estimater.add(Dense(5,activation='relu', input_dim=7))


estimater.add(Dense(5,activation='relu'))

estimater.add(Dense(2,activation='relu'))
 
estimater.compile(optimizer='Adam',loss='mean_absolute_percentage_error',metrics=['mean_absolute_error'])

estimater.fit(X_train,bagimli, epochs=500)


estimatedteam=estimater.predict(X_test1)

estimatedteam=pd.DataFrame(data=estimatedteam,index=range(len(estimatedteam)),columns=['ORtg','DRtg'])


estimatedteam.to_csv('estimatedteam.csv')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    