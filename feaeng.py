import pandas as pd

import numpy as np

from deneme1 import TotalPlayerAdvanced
from deneme1 import TotalPlayerpergame
from deneme1 import TotalPlayerper100Pos


from deneme1 import playeradvanced2022
from deneme1 import playerpergame2022
from deneme1 import playerper1002022


frames=[TotalPlayerAdvanced,TotalPlayerpergame,TotalPlayerper100Pos]     
TotalPlayerData=pd.concat(frames,axis=1)

TotalPlayerData=TotalPlayerData.loc[:,~TotalPlayerData.T.duplicated()]   ### Cleaning duplicate data
 
MpPData=TotalPlayerData[TotalPlayerData["MP"].values>=160]
MpPData=MpPData.replace(np.nan, 0)
MpPData.info()

# corrP=MpPData.corr()
MpPData=MpPData.reset_index()

MpPData=MpPData.drop(columns=['level_0','level_1','Rk','Pos1'])

MpPData=MpPData.rename({'G▼':'G','PF':'PerF'},axis=1)

DataForTrain=MpPData.drop(columns=['FG','FGA','3P','3PA','2P','2PA','FT','FTA','TS%','FG%','3P%','2P%','FT%','Player','Tm','eFG%'])

DataForTrain['Pos'].replace(['SG-SF','C-PF','PF-C','SG-PG','PG-SG','SF-PF','PF-SF','SF-SG','SF-C'],['SG','C','PF','SG','PG','SF','PF','SF','SF'],inplace=True) ### Organize pozisitons 

DataforTrain1=DataForTrain.iloc[:,0]

# Collecting Test data

testdatas=[playeradvanced2022,playerpergame2022,playerper1002022]   
Totaltestdata=pd.concat(testdatas,axis=1)
Totaltestdata=Totaltestdata.loc[:,~Totaltestdata.T.duplicated()]
Totaltestdata=Totaltestdata.replace(np.nan, 0)
MpPTestData=Totaltestdata[Totaltestdata["MP"].values>=130]

MpPTestData=MpPTestData.reset_index()

MpPTestData=MpPTestData.drop(columns=['index','Rk',])

MpPTestData=MpPTestData.rename({'G▼':'G','PF':'PerF'},axis=1)
DataForTest=MpPTestData.drop(columns=['FG','FGA','3P','3PA','2P','2PA','FT','FTA','TS%','FG%','3P%','2P%','FT%','Player','Tm','eFG%'])
DataForTest['Pos'].replace(['SG-SF','C-PF','PF-C','SG-PG','PG-SG','SF-PF','PF-SF','SF-SG','SF-C'],['SG','C','PF','SG','PG','SF','PF','SF','SF'],inplace=True) ### Organize pozisitons

Testpozisition=DataForTest.iloc[:,0]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
DataForTrain1 = le.fit_transform(DataforTrain1)

Testpozisition = le.fit_transform(Testpozisition)

DataForTrain1 = pd.DataFrame(data=DataForTrain1, index = range(len(DataForTrain1)), columns = ['Posi'])

Testpozisition = pd.DataFrame(data=Testpozisition, index = range(len(Testpozisition)), columns = ['Posi'])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
#                         remainder="passthrough")

ohe1=OneHotEncoder()
DataForTrain1 = pd.DataFrame(ohe1.fit_transform(DataForTrain1[['Posi']]).toarray())

Testpozisition = pd.DataFrame(ohe1.fit_transform(Testpozisition[['Posi']]).toarray())

DataForTrain1.columns=['C','PF','PG','SF','SG','no']

Testpozisition.columns=['C','PF','PG','SF','SG']
# DataForTrain1=ohe1.fit_transform(DataForTrain1)

Totaldatafortrain=pd.concat([DataForTrain1,DataForTrain],axis=1)
Totaldatafortrain=Totaldatafortrain.drop(['Pos','no'],axis=1)

Totaldatafortest=pd.concat([Testpozisition,DataForTest],axis=1)

Totaldatafortest=Totaldatafortest.drop(['Pos'],axis=1)

# Data preprocessing ###


Totaldatafortrain.info()
independenttraindata=Totaldatafortrain.loc[:,['C','PF','PG','SF','SG','Age','MP','USG%','MPG','G','GS']]
dependenttraindata=Totaldatafortrain.loc[:,['AST','PTS','TRB','TOV']]

independenttestdata=Totaldatafortest.loc[:,['C','PF','PG','SF','SG','Age','MP','USG%','MPG','G','GS']]
dependenttestdata=Totaldatafortest.loc[:,['AST','PTS','TRB','TOV']]

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(independenttraindata,dependenttraindata,test_size=0.3, random_state=0)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
X_train = scaler.fit_transform(independenttraindata)
X_test1 = scaler.fit_transform(independenttestdata)

### Deep Learning ###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

estimater=Sequential()

estimater.add(Dense(10,activation='tanh', input_dim=11))

estimater.add(Dense(10,activation='relu'))

estimater.add(Dense(10,activation='relu'))

estimater.add(Dense(4,activation='relu'))
 
estimater.compile(optimizer='Adam',loss='mean_absolute_percentage_error',metrics=['mean_absolute_error'])

estimater.fit(X_train,dependenttraindata, epochs=100)


estimatedplayer2022=estimater.predict(X_test1)


estimatedplayer2021=estimatedplayer2022[:,0]

xtest2=dependenttestdata.iloc[:,0]

# # print(accuracy_score(estimatedplayer2022, dependenttestdata))

# # randomForest_cm = confusion_matrix(estimatedplayer2022,dependenttestdata,normalize=('true'))
estimatedplayer2022=pd.DataFrame(data=estimatedplayer2022,index=range(len(estimatedplayer2022)),columns=["AST","PTS","TRB","TOV"])


player2022=MpPTestData.loc[:,['Player','Pos','Age','Tm','G','MP','GS','MPG']]

predictedplayer=pd.concat([player2022,estimatedplayer2022],axis=1)


predictedplayer.to_csv('predictedplayer1.csv')