import csv
import requests
import pandas as pd
import urllib.parse
import json
import datetime

# Function to get team at kickoff time
def get_team_at_kickoff(transfers, kickoff_date):
    current_team = transfers[0]["to"]["clubName"]
    for transfer in transfers:
        transfer_date = datetime.datetime.strptime(transfer["date"], "%b %d, %Y")
        kickoff_time = datetime.datetime.strptime(kickoff_date, "%Y-%m-%dT%H:%M:%SZ")

        print(f"transfer_date: {transfer_date} and kickoff time: {kickoff_time}")
        if transfer_date > kickoff_time:
            current_team = transfer["from"]["clubName"]
        else:
            return current_team
    return current_team

# Load CSV data
def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Process CSV data
def process_csv(data):
    # Filter out rows with team_x filled
    data = data[data["team_x"].isnull()]
    
    # Get unique player names
    player_names = data["name"].unique()
    
    # Get player IDs
    transfers = {}
    try:
        with open('transfers.json', 'r') as f:
            transfers=json.load(f)
    except Exception:
        pass
        
    
    # Fill team_x field
    for index, row in data.iterrows():
        kickoff_time = row["kickoff_time"]
        player_name = row["name"]
        if player_name in transfers:
            team = get_team_at_kickoff(transfers[player_name], kickoff_time)
            if team:
                data.at[index, "team_x"] = team
    
    return data

# Save CSV data
def save_csv(data, file_path):
    data.to_csv(file_path, index=False)

# Main function
def main():
    file_path = "data/cleaned_merged_seasons.csv"  # replace with your CSV file path
    data = load_csv(file_path)
    data = process_csv(data)
    save_csv(data, "cleaned_merged_seasons_with_teams.csv")

if __name__ == "__main__":
    main()