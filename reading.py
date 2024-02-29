import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_player_data():
    """Load player data from multiple seasons."""
    # Reading player advanced data
    playeradvanced2022 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playeradvanced.xlsx")
    playeradvanced2021 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2020-2021\playeradvanced.xlsx")
    playeradvanced2020 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2019-2020\playeradvanced.xlsx")
    playeradvanced2019 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2018-2019\playeradvanced2.xlsx")
    playeradvanced2018 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2017-2018\playeradvanced.xlsx")
    playeradvanced2017 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2016-2017\playeradvanced.xlsx")

    # Concatenating player advanced data from multiple seasons
    frames_adv = [playeradvanced2021, playeradvanced2020, playeradvanced2019, playeradvanced2018, playeradvanced2017]
    TotalPlayerAdvanced = pd.concat(frames_adv, keys=("2021", "2020", "2019", "2018", "2017"))

    return TotalPlayerAdvanced

def load_team_data():
    """Load team data from multiple seasons."""
    # Reading team advanced data
    teamadvanced2022 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2021-2022\teamadvanced.xlsx")
    teamadvanced2021 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2020-2021\teamadvanced.xlsx")
    teamadvanced2020 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2019-2020\teamadvanced.xlsx")
    teamadvanced2019 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2018-2019\teamadvanced.xlsx")
    teamadvanced2018 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2017-2018\teamadvanced.xlsx")
    teamadvanced2017 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2016-2017\teamadvanced.xlsx")

    # Concatenating team advanced data from multiple seasons
    frames_team_adv = [teamadvanced2021, teamadvanced2020, teamadvanced2019, teamadvanced2018, teamadvanced2017]
    TotalTeamAdvanced = pd.concat(frames_team_adv)

    return TotalTeamAdvanced

def preprocess_team_data(team_advanced, team_per_game):
    """Preprocess team data by sorting and dropping unnecessary columns."""
    # Sorting values by team name
    team_advanced = team_advanced.sort_values(by=['Team']).reset_index().drop(columns=(['index', 'Rk']))
    team_per_game = team_per_game.sort_values(by=['Team']).reset_index().drop(columns=(['index', 'Rk']))
    # Dropping league average rows
    team_advanced = team_advanced[team_advanced.Team != 'League Average']
    team_per_game = team_per_game[team_per_game.Team != 'League Average']

    return team_advanced, team_per_game

def merge_player_data(player_per_game, player_advanced, player_per_100):
    """Merge player dataframes."""
    # Selecting relevant columns from player dataframes
    player_per_game = player_per_game.rename({'MP':'MPG', 'Pos':'Pos1'}, axis=1)
    player_per_100_pos = player_per_100.iloc[:, 29:31]
    # Merging player dataframes
    total_player_data = pd.concat([player_advanced, player_per_game, player_per_100_pos], axis=1)
    total_player_data = total_player_data.loc[:, ~total_player_data.T.duplicated()]

    return total_player_data

def merge_team_data(team_per_game, team_advanced, team_per_game_off, team_advanced_22):
    """Merge team dataframes."""
    # Merging team per game and advanced data
    train_team = pd.concat([team_per_game, team_advanced], axis=1)
    test_team = pd.concat([team_per_game_off, team_advanced_22], axis=1)

    return train_team, test_team

def plot_relationship(x_value, y_value, data, title):
    """Plot relationship between two variables."""
    sns.lmplot(x=x_value, y=y_value, data=data)
    plt.title(title)

def main():
    # Load player data
    TotalPlayerAdvanced = load_player_data()

    # Load team data
    TotalTeamAdvanced = load_team_data()

    # Load team per game data
    team_per_game_off_22 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2021-2022\teampergameoff.xlsx")

    # Preprocess team data
    TotalTeamAdvanced, TotalTeamPerGame = preprocess_team_data(TotalTeamAdvanced, team_per_game_off_22)

    # Merge player data
    TotalPlayerPerGame = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playerpergame.xlsx")
    TotalPlayerPer100 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Player\2021-2022\playerper100.xlsx")
    TotalPlayerData = merge_player_data(TotalPlayerPerGame, TotalPlayerAdvanced, TotalPlayerPer100)

    # Merge team data
    team_advanced_22 = pd.read_excel(r"C:\Users\mehme\OneDrive\Masaüstü\proje\veriler\Team\2021-2022\teamadvanced.xlsx")
    train_team, test_team = merge_team_data(TotalTeamPerGame, TotalTeamAdvanced, team_per_game_off_22, team_advanced_22)

    # Plot relationships
    plot_relationship('W', 'DRtg', team_advanced_22, 'Win Based on Defensive Rating')
    plot_relationship('W', 'ORtg', team_advanced_22, 'Win Based on Offensive Rating')
    plot_relationship('W', 'NRtg', team_advanced_22, 'Win Based on Net Rating')

    plt.show()

if __name__ == "__main__":
    main()
