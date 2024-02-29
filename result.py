import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def calculate_winning_losses_ratio(estimated_team):
    """
    Calculate the winning and losses ratio for each team.

    Args:
    estimated_team (DataFrame): DataFrame containing estimated team data.

    Returns:
    DataFrame: DataFrame with winning and losses ratio for each team.
    """
    teams = estimated_team['Team']
    winning_losses_ratio = pd.DataFrame(columns=['W', 'L'])

    for i in range(len(estimated_team)):
        total_games = 82  # Total number of games in a season
        estimated_wins = 82 * estimated_team.iloc[i, 0] / estimated_team.iloc[i, 2]
        estimated_losses = total_games - estimated_wins
        winning_losses_ratio.loc[i] = [estimated_wins, estimated_losses]

    return winning_losses_ratio

def plot_team_performance(winning_losses_ratio, teams):
    """
    Plot the performance of each team based on their winning and losses ratio.

    Args:
    winning_losses_ratio (DataFrame): DataFrame containing winning and losses ratio for each team.
    teams (list): List of team names.

    Returns:
    None
    """
    sns.scatterplot(x='W', y='L', data=winning_losses_ratio, hue='Team')
    plt.xlabel('Wins')
    plt.ylabel('Losses')
    plt.title('Team Performance')
    plt.show()

def main():
    # Load data
    estimated_team = load_data('estimatedteam.csv')
    estimated_team = estimated_team.drop(columns=['Unnamed: 0'])

    # Calculate winning and losses ratio
    winning_losses_ratio = calculate_winning_losses_ratio(estimated_team)

    # Define team names
    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

    # Plot team performance
    plot_team_performance(winning_losses_ratio, teams)

if __name__ == "__main__":
    main()
