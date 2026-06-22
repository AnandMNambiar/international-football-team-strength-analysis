import pandas as pd

# Load dataset
df = pd.read_csv(
    r"C:\Users\Admin\OneDrive\Desktop\ML Projects\World Cup 26\results.csv"
)

# First look at the data
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDate Range:")
print("Earliest Match:", df['date'].min())
print("Latest Match:", df['date'].max())
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Create year column
df['year'] = df['date'].dt.year

print(df[['date', 'year']].head())
matches_per_year = df['year'].value_counts().sort_index()

print(matches_per_year.tail(20))
print(df['tournament'].value_counts().head(20))
modern_df = df[df['year'] >= 2010].copy()

print("Original Matches:", len(df))
print("Modern Matches:", len(modern_df))
missing_scores = modern_df[
    modern_df['home_score'].isna() |
    modern_df['away_score'].isna()
]

print(missing_scores[['date','home_team','away_team','tournament']].head(20))
def get_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

modern_df['result'] = modern_df.apply(get_result, axis=1)

print(modern_df['result'].value_counts())
train_df = modern_df.dropna(subset=['home_score', 'away_score']).copy()

print("Training Matches:", len(train_df))
print("Removed Future Matches:", len(modern_df) - len(train_df))
home_wins = train_df[train_df['result'] == 'Home Win']['home_team']
away_wins = train_df[train_df['result'] == 'Away Win']['away_team']

all_wins = pd.concat([home_wins, away_wins])

team_wins = all_wins.value_counts()

print(team_wins.head(20))
# Total matches played by each team

home_matches = train_df['home_team'].value_counts()
away_matches = train_df['away_team'].value_counts()

total_matches = home_matches.add(away_matches, fill_value=0)

# Total wins already calculated
team_wins = all_wins.value_counts()

# Win percentage
win_percentage = (team_wins / total_matches * 100).sort_values(ascending=False)

print(win_percentage.head(20))
eligible_teams = total_matches[total_matches >= 50].index

filtered_win_pct = win_percentage.loc[
    win_percentage.index.intersection(eligible_teams)
]

print(filtered_win_pct.head(20))
# Goals scored by each team

home_goals_scored = train_df.groupby('home_team')['home_score'].sum()
away_goals_scored = train_df.groupby('away_team')['away_score'].sum()

goals_scored = home_goals_scored.add(away_goals_scored, fill_value=0)

# Goals conceded

home_goals_conceded = train_df.groupby('home_team')['away_score'].sum()
away_goals_conceded = train_df.groupby('away_team')['home_score'].sum()

goals_conceded = home_goals_conceded.add(away_goals_conceded, fill_value=0)

# Goal difference

goal_difference = goals_scored - goals_conceded

print(goal_difference.sort_values(ascending=False).head(20))
team_strength = pd.DataFrame()

team_strength['Win_Percentage'] = filtered_win_pct
team_strength['Goal_Difference'] = goal_difference

team_strength = team_strength.fillna(0)

# Normalize both metrics
team_strength['Win_Percentage_Norm'] = (
    team_strength['Win_Percentage']
    / team_strength['Win_Percentage'].max()
)

team_strength['Goal_Difference_Norm'] = (
    team_strength['Goal_Difference']
    / team_strength['Goal_Difference'].max()
)

# Simple strength score
team_strength['Strength_Score'] = (
    0.5 * team_strength['Win_Percentage_Norm']
    + 0.5 * team_strength['Goal_Difference_Norm']
)

team_strength = team_strength.sort_values(
    by='Strength_Score',
    ascending=False
)

print(team_strength.head(20))
import matplotlib.pyplot as plt

top10 = team_strength.head(10)

plt.figure(figsize=(10,5))
plt.bar(top10.index, top10['Strength_Score'])
plt.xticks(rotation=45)
plt.title("Top 10 Team Strength Scores (2010-2026)")
plt.tight_layout()

plt.savefig(
    r"C:\Users\Admin\OneDrive\Desktop\ML Projects\World Cup 26\team_strength_top10.png"
)

print("Chart saved as team_strength_top10.png")
recent_df = df[df['year'] >= 2018].copy()

print("Recent Matches:", len(recent_df))
import os

print(os.getcwd())
