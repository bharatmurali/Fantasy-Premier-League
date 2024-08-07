import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split

# Function to load cleaned_merged_seasons.csv and master_team_list.csv
def load_data(season_file='data/cleaned_merged_seasons.csv', team_file='data/master_team_list.csv'):
    data = pd.read_csv(season_file, encoding='latin1')
    team_data = pd.read_csv(team_file, encoding='latin1')
    return data, team_data

# Function to preprocess the data
def preprocess_data(data, team_data):
    # Convert 'kickoff_time' to datetime
    data['kickoff_time'] = pd.to_datetime(data['kickoff_time'], format='%Y-%m-%dT%H:%M:%SZ')

    # Extract features from 'kickoff_time'
    data['kickoff_month'] = data['kickoff_time'].dt.month
    data['kickoff_hour'] = data['kickoff_time'].dt.hour
    
    # Drop the original 'kickoff_time' column
    data = data.drop(columns=['kickoff_time', 'opp_team_name'])

    # Ensure 'opponent_team' is filled
    data = data[data['opponent_team'] != 0]

    # Convert 'team_x' to team ID using 'master_team_list.csv'
    team_dict = team_data.set_index('team_name')['team'].to_dict()
    data['team_x'] = data['team_x'].map(team_dict)

    # Convert missing values to 0
    data = data.fillna(0)

    # Map 'position' field to digit values
    position_map = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    data['position'] = data['position'].map(position_map)

    return data

def calculate_form_features(data, window_size=5):
    # Sort the data
    data = data.sort_values(['name', 'season_x', 'GW'])
    
    # Function to calculate rolling mean and align index
    def rolling_mean_aligned(group, column):
        return group[column].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate player form
    data['player_form'] = data.groupby(['name', 'season_x', 'GW']).apply(rolling_mean_aligned, 'total_points').reset_index(level=[0,1], drop=True)
    
    # Calculate team offensive and defensive form
    data['team_offensive_form'] = data.groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['team_defensive_form'] = data.groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    
    # Calculate opposition offensive and defensive form
    data['opp_offensive_form'] = data.groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['opp_defensive_form'] = data.groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    
    # Calculate player home and away forms
    data['player_home_form'] = data[data['was_home']].groupby(['name', 'season_x', 'GW']).apply(rolling_mean_aligned, 'total_points').reset_index(level=[0,1], drop=True)
    data['player_away_form'] = data[~data['was_home']].groupby(['name', 'season_x', 'GW']).apply(rolling_mean_aligned, 'total_points').reset_index(level=[0,1], drop=True)
    
    # Calculate team home and away offensive and defensive forms
    data['team_home_offensive_form'] = data[data['was_home']].groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['team_home_defensive_form'] = data[data['was_home']].groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    data['team_away_offensive_form'] = data[~data['was_home']].groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['team_away_defensive_form'] = data[~data['was_home']].groupby(['team_x', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    
    # Calculate opposition home and away offensive and defensive forms
    data['opp_home_offensive_form'] = data[~data['was_home']].groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['opp_home_defensive_form'] = data[~data['was_home']].groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    data['opp_away_offensive_form'] = data[data['was_home']].groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_scored').reset_index(level=[0,1], drop=True)
    data['opp_away_defensive_form'] = data[data['was_home']].groupby(['opponent_team', 'season_x']).apply(rolling_mean_aligned, 'goals_conceded').reset_index(level=[0,1], drop=True)
    
    # Fill NaN values with 0
    form_columns = [
        'player_form', 
        'team_offensive_form', 'team_defensive_form', 
        'opp_offensive_form', 'opp_defensive_form', 
        'player_home_form', 'player_away_form', 
        'team_home_offensive_form', 'team_home_defensive_form', 
        'team_away_offensive_form', 'team_away_defensive_form', 
        'opp_home_offensive_form', 'opp_home_defensive_form', 
        'opp_away_offensive_form', 'opp_away_defensive_form'
    ]
    data[form_columns] = data[form_columns].fillna(0)
    
    return data

# Function to train and predict using rolling windows for a single season
def rolling_window_predict_player_season(player_season_data, window_size=5):
    results = []
    for i in range(window_size, len(player_season_data)):
        train = player_season_data.iloc[i-window_size:i]
        test = player_season_data.iloc[i:i+1]

        X_train = train.drop(columns=['total_points', 'name', 'GW', 'season_x'])
        y_train = train['total_points']
        X_test = test.drop(columns=['total_points', 'name', 'GW', 'season_x'])
        y_test = test['total_points']

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'name': player_season_data['name'].values[0],
            'GW': test['GW'].values[0],
            'season_x': test['season_x'].values[0],
            'actual': y_test.values[0],
            'predicted': y_pred[0]
        })
    return results

# Load data
data, team_data = load_data()

# Preprocess data
data = preprocess_data(data, team_data)

# Calculate form features
data = calculate_form_features(data)

print("PREPROCESSED DATA\n")
print(data)

# Split data into training and validation sets
last_season = data['season_x'].max()
validation_data = data[data['season_x'] == last_season]
training_data = data[data['season_x'] < last_season]

print("VALIDATION DATA\n")
print(validation_data)

print("TRAINING DATA\n")
print(training_data)

# Train model and make predictions using rolling window for each season independently
seasons = training_data['season_x'].unique()
results = []

for season in seasons:
    season_data = training_data[training_data['season_x'] == season]
    players = season_data['name'].unique()
    season_results = Parallel(n_jobs=-1)(delayed(rolling_window_predict_player_season)(season_data[season_data['name'] == player]) for player in players)
    results.extend([item for sublist in season_results for item in sublist])

predictions = pd.DataFrame(results)

# Validate model on the validation dataset
X_val = validation_data.drop(columns=['total_points', 'name', 'GW', 'season_x'])
y_val = validation_data['total_points']

# Incorporate hyperparameter tuning
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_val, y_val)
best_model = grid_search.best_estimator_

# Predict on validation set
val_predictions = best_model.predict(X_val)
val_results = {
    'actual': y_val,
    'predicted': val_predictions
}
val_predictions_df = pd.DataFrame(val_results)

# Calculate performance metrics on validation data
mse = mean_squared_error(val_predictions_df['actual'], val_predictions_df['predicted'])
mae = mean_absolute_error(val_predictions_df['actual'], val_predictions_df['predicted'])
rmse = np.sqrt(mse)

# Save the trained model
dump(best_model, 'random_forest_model.joblib')

print(f'Mean Squared Error (Validation): {mse}')
print(f'Mean Absolute Error (Validation): {mae}')
print(f'Root Mean Squared Error (Validation): {rmse}')

# Filter data where actual points are greater than 10
high_score_indices = y_val > 10
y_val_high = y_val[high_score_indices]
val_predictions_high = val_predictions[high_score_indices]

# Calculate metrics for high scores
mse_high = mean_squared_error(y_val_high, val_predictions_high)
mae_high = mean_absolute_error(y_val_high, val_predictions_high)
rmse_high = np.sqrt(mse_high)

print(f"Mean Squared Error (High Scores): {mse_high}")
print(f"Mean Absolute Error (High Scores): {mae_high}")
print(f"Root Mean Squared Error (High Scores): {rmse_high}")

# Save predictions to a CSV file
predictions.to_csv('predictions.csv', index=False)
val_predictions_df.to_csv('validation_predictions.csv', index=False)