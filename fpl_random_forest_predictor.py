import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

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

# Function to train and predict using rolling windows
def rolling_window_predict_player(player_data, window_size=5):
    results = []
    for i in range(window_size, len(player_data)):
        train = player_data.iloc[i-window_size:i]
        test = player_data.iloc[i:i+1]

        X_train = train.drop(columns=['total_points', 'name', 'GW', 'season_x'])
        y_train = train['total_points']
        X_test = test.drop(columns=['total_points', 'name', 'GW', 'season_x'])
        y_test = test['total_points']

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'name': player_data['name'].values[0],
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

# Split data into training and validation sets
last_season = data['season_x'].max()
validation_data = data[data['season_x'] == last_season]
training_data = data[data['season_x'] < last_season]

# Train model and make predictions using rolling window
players = training_data['name'].unique()
results = Parallel(n_jobs=-1)(delayed(rolling_window_predict_player)(training_data[training_data['name'] == player]) for player in players)

# Flatten the list of results
results = [item for sublist in results for item in sublist]
predictions = pd.DataFrame(results)

# Validate model on the validation dataset
X_val = validation_data.drop(columns=['total_points', 'name', 'GW', 'season_x'])
y_val = validation_data['total_points']

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_val, y_val)

best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'best_random_forest_regressor_model.joblib')

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
