import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data():
    """
    Load data from CSV files.

    Returns:
    Predictedplayers, Predictedplayers1, trainteam, testteam, totalplayerdata: DataFrames containing the required data.
    """
    Predictedplayers = pd.read_csv('predictedplayers.csv')
    Predictedplayers1 = pd.read_csv('predictedplayer1.csv')
    trainteam = pd.read_csv('trainteam.csv')
    testteam = pd.read_csv('testteam.csv')
    totalplayerdata = pd.read_csv('totalplayerdata.csv')
    return Predictedplayers, Predictedplayers1, trainteam, testteam, totalplayerdata

def calculate_averages(data, Teams, Columns):
    """
    Calculate average player statistics and team ratings.

    Args:
    data: DataFrame containing player statistics.
    Teams: List of team abbreviations.
    Columns: List of column names.

    Returns:
    sumdata, meandata: DataFrames containing summed and averaged statistics.
    """
    sumdata = pd.DataFrame(columns=Columns)
    meandata = pd.DataFrame(columns=["ORtg", "DRtg"])
    
    for team in Teams:
        team_data = data.loc[data['Tm'] == team]
        mpg_filtered_data = team_data.loc[team_data["MPG"] >= 28]
        players = team_data['Player'].head(9)
        team_ratings = data.loc[data['Player'].isin(players)]
        mean_ratings = team_ratings.groupby('Player', as_index=False).mean()
        
        for column in Columns:
            if column == 'ORtg' or column == 'DRtg':
                meandata.loc[team, column] = mean_ratings[column].mean()
            else:
                sumdata.loc[team, column] = team_data[column].head(9).sum()
    
    return sumdata, meandata

def plot_value(x_value, y_value, data, hue):
    """
    Plot a scatterplot.

    Args:
    x_value: Name of the column for the x-axis.
    y_value: Name of the column for the y-axis.
    data: DataFrame containing the data.
    hue: Hue variable.

    Returns:
    None
    """
    sns.scatterplot(x=x_value, y=y_value, data=data, hue=hue)
    plt.title(x_value + ' to ' + y_value)
    plt.show()

def calculate_difference(a, b):
    """
    Calculate the difference between two values.

    Args:
    a: First value.
    b: Second value.

    Returns:
    Difference between a and b.
    """
    return b - a

def main():
    # Load data
    Predictedplayers, Predictedplayers1, trainteam, testteam, totalplayerdata = load_data()

    # Define Teams and Columns
    Teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    Columns = ['AST', 'PTS', 'TRB', 'ORtg', 'DRtg', 'TOV']

    # Calculate averages
    sumdata, meandata = calculate_averages(Predictedplayers1, Teams, Columns)

    # Plot value
    plot_value('G', 'ORTg', Predictedplayers, 'Tm')

    # Calculate correlation matrix
    corrMatrix = Predictedplayers.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

    # Calculate team ratings difference
    Totaldata = pd.DataFrame(Teams, columns=['Team'])
    Totaldata['NRtg'] = Totaldata.apply(lambda x: calculate_difference(x['DRtg'], x['ORtg']), axis=1)

    # Prepare training and testing data
    bagimli = trainteam.loc[:, ['W', 'L']]
    bagimsiz = trainteam.loc[:, ['AST', 'PTS', 'TRB', 'TOV', 'ORtg', 'DRtg', 'NRtg']]
    X_train, X_test, y_train, y_test = train_test_split(bagimsiz, bagimli, test_size=0.3, random_state=0)

    # Scale data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Build and train the model
    model = Sequential()
    model.add(Dense(5, activation='relu', input_dim=7))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='Adam', loss='mean_absolute_percentage_error', metrics=['mean_absolute_error'])
    model.fit(X_train_scaled, y_train, epochs=500)

    # Predict team ratings
    tester = Totaldata.loc[:, ['AST', 'PTS', 'TRB', 'TOV', 'ORtg', 'DRtg', 'NRtg']]
    X_test1_scaled = scaler.fit_transform(tester)
    estimated_team = model.predict(X_test1_scaled)
    estimated_team = pd.DataFrame(data=estimated_team, index=range(len(estimated_team)), columns=['ORtg', 'DRtg'])
    estimated_team.to_csv('estimatedteam.csv')

if __name__ == "__main__":
    main()
