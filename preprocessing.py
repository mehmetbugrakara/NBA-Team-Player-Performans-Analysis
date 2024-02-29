from deneme1 import TotalPlayerAdvanced
from deneme1 import TotalPlayerpergame
from deneme1 import TotalPlayerper100Pos
from deneme1 import playeradvanced2022
from deneme1 import playerpergame2022
from deneme1 import playerper1002022

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing

def clean_and_prepare_data(TotalPlayerAdvanced, TotalPlayerpergame, TotalPlayerper100Pos, playeradvanced2022, playerpergame2022, playerper1002022):
    """
    Combines, cleans, and prepares player data for training and testing.

    Args:
    TotalPlayerAdvanced, TotalPlayerpergame, TotalPlayerper100Pos: DataFrames containing player data.
    playeradvanced2022, playerpergame2022, playerper1002022: DataFrames containing test player data for 2022.

    Returns:
    Totaldatafortrain, Totaldatafortest: Prepared training and testing data.
    """
    # Concatenate player data
    frames = [TotalPlayerAdvanced, TotalPlayerpergame, TotalPlayerper100Pos]     
    TotalPlayerData = pd.concat(frames, axis=1)

    # Remove duplicate columns
    TotalPlayerData = TotalPlayerData.loc[:, ~TotalPlayerData.T.duplicated()]

    # Filter players with minimum playing time
    MpPData = TotalPlayerData[TotalPlayerData["MP"].values >= 160]
    MpPData = MpPData.replace(np.nan, 0)

    # Reset index and drop unnecessary columns
    MpPData = MpPData.reset_index()
    MpPData = MpPData.drop(columns=['level_0', 'level_1', 'Rk', 'Pos1'])

    # Rename columns
    MpPData = MpPData.rename({'G▼': 'G', 'PF': 'PerF'}, axis=1)

    # Prepare data for training
    DataForTrain = MpPData.drop(columns=['FG','FGA','3P','3PA','2P','2PA','FT','FTA','TS%','FG%','3P%','2P%','FT%','Player','Tm','eFG%'])
    DataForTrain['Pos'].replace(['SG-SF','C-PF','PF-C','SG-PG','PG-SG','SF-PF','PF-SF','SF-SG','SF-C'],['SG','C','PF','SG','PG','SF','PF','SF','SF'], inplace=True)

    DataforTrain1 = DataForTrain.iloc[:, 0]

    # Collect Test data
    testdatas = [playeradvanced2022, playerpergame2022, playerper1002022]   
    Totaltestdata = pd.concat(testdatas, axis=1)
    Totaltestdata = Totaltestdata.loc[:, ~Totaltestdata.T.duplicated()]
    Totaltestdata = Totaltestdata.replace(np.nan, 0)
    MpPTestData = Totaltestdata[Totaltestdata["MP"].values >= 130]

    # Reset index and drop unnecessary columns
    MpPTestData = MpPTestData.reset_index()
    MpPTestData = MpPTestData.drop(columns=['index', 'Rk', ])

    MpPTestData = MpPTestData.rename({'G▼': 'G', 'PF': 'PerF'}, axis=1)
    DataForTest = MpPTestData.drop(columns=['FG','FGA','3P','3PA','2P','2PA','FT','FTA','TS%','FG%','3P%','2P%','FT%','Player','Tm','eFG%'])
    DataForTest['Pos'].replace(['SG-SF','C-PF','PF-C','SG-PG','PG-SG','SF-PF','PF-SF','SF-SG','SF-C'],['SG','C','PF','SG','PG','SF','PF','SF','SF'], inplace=True)

    Testpozisition = DataForTest.iloc[:, 0]

    # Encode categorical variables
    le = preprocessing.LabelEncoder()
    DataForTrain1 = le.fit_transform(DataforTrain1)
    Testpozisition = le.fit_transform(Testpozisition)
    DataForTrain1 = pd.DataFrame(data=DataForTrain1, index=range(len(DataForTrain1)), columns=['Posi'])
    Testpozisition = pd.DataFrame(data=Testpozisition, index=range(len(Testpozisition)), columns=['Posi'])

    # One-hot encode categorical variables
    ohe1 = OneHotEncoder()
    DataForTrain1 = pd.DataFrame(ohe1.fit_transform(DataForTrain1[['Posi']]).toarray())
    Testpozisition = pd.DataFrame(ohe1.fit_transform(Testpozisition[['Posi']]).toarray())
    DataForTrain1.columns = ['C', 'PF', 'PG', 'SF', 'SG', 'no']
    Testpozisition.columns = ['C', 'PF', 'PG', 'SF', 'SG']

    # Concatenate encoded variables with data
    Totaldatafortrain = pd.concat([DataForTrain1, DataForTrain], axis=1)
    Totaldatafortrain = Totaldatafortrain.drop(['Pos', 'no'], axis=1)
    Totaldatafortest = pd.concat([Testpozisition, DataForTest], axis=1)
    Totaldatafortest = Totaldatafortest.drop(['Pos'], axis=1)

    return Totaldatafortrain, Totaldatafortest

def scale_data(train_data, test_data):
    """
    Scales the training and testing data using RobustScaler.

    Args:
    train_data: Training data.
    test_data: Testing data.

    Returns:
    scaled_train_data, scaled_test_data: Scaled training and testing data.
    """
    scaler = RobustScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.fit_transform(test_data)
    return scaled_train_data, scaled_test_data

def build_and_train_model(train_data, train_labels):
    """
    Builds and trains a deep learning model using Keras.

    Args:
    train_data: Scaled training data.
    train_labels: Labels for the training data.

    Returns:
    model: Trained deep learning model.
    """
    model = Sequential()
    model.add(Dense(10, activation='tanh', input_dim=train_data.shape[1]))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.compile(optimizer='Adam', loss='mean_absolute_percentage_error', metrics=['mean_absolute_error'])
    model.fit(train_data, train_labels, epochs=100)
    return model

def predict_and_save(model, test_data, test_player_data):
    """
    Predicts player statistics for 2022 and saves the results to a CSV file.

    Args:
    model: Trained deep learning model.
    test_data: Scaled testing data.
    test_player_data: DataFrame containing player information for 2022.
    """
    predicted_stats = model.predict(test_data)
    predicted_stats = pd.DataFrame(data=predicted_stats, index=range(len(predicted_stats)), columns=["AST", "PTS", "TRB", "TOV"])
    player_info = test_player_data.loc[:, ['Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'GS', 'MPG']]
    predicted_player = pd.concat([player_info, predicted_stats], axis=1)
    predicted_player.to_csv('predicted_player_2022.csv', index=False)

def main():
    # Load data
    TotalPlayerAdvanced = ...
    TotalPlayerpergame = ...
    TotalPlayerper100Pos = ...
    playeradvanced2022 = ...
    playerpergame2022 = ...
    playerper1002022 = ...

    # Clean and prepare data
    Totaldatafortrain, Totaldatafortest = clean_and_prepare_data(TotalPlayerAdvanced, TotalPlayerpergame, TotalPlayerper100Pos, playeradvanced2022, playerpergame2022, playerper1002022)

    # Split data into train and test sets
    independenttraindata = Totaldatafortrain.loc[:, ['C', 'PF', 'PG', 'SF', 'SG', 'Age', 'MP', 'USG%', 'MPG', 'G', 'GS']]
    dependenttraindata = Totaldatafortrain.loc[:, ['AST', 'PTS', 'TRB', 'TOV']]
    independenttestdata = Totaldatafortest.loc[:, ['C', 'PF', 'PG', 'SF', 'SG', 'Age', 'MP', 'USG%', 'MPG', 'G', 'GS']]

    # Scale data
    scaled_train_data, scaled_test_data = scale_data(independenttraindata, independenttestdata)

    # Train model
    model = build_and_train_model(scaled_train_data, dependenttraindata)

    # Predict and save results
    predict_and_save(model, scaled_test_data, Totaldatafortest)

if __name__ == "__main__":
    main()


