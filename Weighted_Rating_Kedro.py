# import necessary libraries

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataSet
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.runner import SequentialRunner
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Define your functions for data processing
def load_data():
    return pd.read_csv("D:\Data Science\output_file_1_1000.csv")

def preprocess_data(df):
    # Data preprocessing steps
    return df

def calculate_batsman_stats(df):
    batsman_runs = df.groupby(['batter_id', 'event_id', 'innings_id'])['batter_runs'].max().groupby('batter_id').sum()

    batsman_balls = df.groupby(['batter_id', 'event_id', 'innings_id'])['batter_balls_faced'].max().groupby('batter_id').sum()

    batsman_outs = df[df['outcome'] == 'out'].groupby('batter_id').size()

    batsman_max_score = df.groupby(['batter_id', 'event_id', 'innings_id'])['batter_runs'].max().groupby('batter_id').max()

    batsman_fours = df[df['outcome'] == 'four'].groupby(['batter_id', 'event_id', 'innings_id']).size().groupby('batter_id').sum()

    batsman_sixes = df[df['outcome'] == 'six'].groupby(['batter_id', 'event_id', 'innings_id']).size().groupby('batter_id').sum()

    batsman_fifties = df.groupby(['batter_id', 'event_id', 'innings_id'])['batter_runs'].max().groupby('batter_id').apply(lambda x: sum((x >= 50) & (x < 100)))

    batsman_hundreds = df.groupby(['batter_id', 'event_id', 'innings_id'])['batter_runs'].max().groupby('batter_id').apply(lambda x: sum(x >= 100))

    batsman_stats = pd.concat([batsman_runs, batsman_balls, batsman_outs, batsman_max_score, batsman_fours, batsman_sixes, batsman_fifties, batsman_hundreds], axis=1)

    batsman_stats.columns = ['total_runs', 'total_balls', 'times_out', 'max_score', 'num_fours', 'num_sixes', 'num_fifties', 'num_hundreds']

    batsman_stats['average'] = batsman_stats['total_runs'] / batsman_stats['times_out']

    batsman_stats['strike_rate'] = batsman_stats['total_runs'] / batsman_stats['total_balls'] * 100

    scaler = MinMaxScaler()

    # Normalize the batsman statistics
    batsman_stats_normalized = pd.DataFrame(scaler.fit_transform(batsman_stats), columns=batsman_stats.columns, index=batsman_stats.index)

    # Calculate the overall rating for batters and bowlers as the mean of the normalized metrics
    batsman_stats_normalized['rating'] = batsman_stats_normalized.mean(axis=1)
    return batsman_stats_normalized

def calculate_bowler_stats(df):

    bowler_innings = df.groupby('bowler_id')['innings_id'].nunique()

    bowler_runs_conceded = df.groupby(['bowler_id', 'event_id', 'innings_id'])['bowler_conceded'].max().groupby('bowler_id').sum()

    bowler_wickets = df[df['outcome'] == 'out'].groupby('bowler_id').size()

    bowler_max_wickets = df[df['outcome'] == 'out'].groupby(['bowler_id', 'event_id', 'innings_id']).size().groupby('bowler_id').max()

    bowler_overs = df.groupby(['bowler_id', 'event_id', 'innings_id'])['bowler_overs'].max().groupby('bowler_id').sum()

    bowler_three_wickets = df[df['outcome'] == 'out'].groupby(['bowler_id', 'event_id', 'innings_id']).size().groupby('bowler_id').apply(lambda x: sum(x >= 3))

    bowler_five_wickets = df[df['outcome'] == 'out'].groupby(['bowler_id', 'event_id', 'innings_id']).size().groupby('bowler_id').apply(lambda x: sum(x >= 5))

    bowler_stats = pd.concat([bowler_innings, bowler_runs_conceded, bowler_wickets, bowler_max_wickets, bowler_overs, bowler_three_wickets, bowler_five_wickets], axis=1)

    bowler_stats.columns = ['innings', 'total_runs_conceded', 'total_wickets', 'max_wickets_in_match', 'total_overs', 'num_three_wickets', 'num_five_wickets']

    bowler_stats['average'] = bowler_stats['total_runs_conceded'] / bowler_stats['total_wickets']

    bowler_stats['economy'] = bowler_stats['total_runs_conceded'] / bowler_stats['total_overs']

    scaler = MinMaxScaler()
    # Normalize the bowler statistics
    bowler_stats_normalized = pd.DataFrame(scaler.fit_transform(bowler_stats), columns=bowler_stats.columns, index=bowler_stats.index)

    # Calculate the overall rating for batters and bowlers as the mean of the normalized metrics
    bowler_stats_normalized['rating'] = bowler_stats_normalized.mean(axis=1)

    return bowler_stats_normalized

def calculate_ratings(batsman_stats_normalized, bowler_stats_normalized):
    # Calculate the overall rating for batters and bowlers as the mean of the normalized metrics
    batsman_stats_normalized['rating'] = batsman_stats_normalized.mean(axis=1)
    bowler_stats_normalized['rating'] = bowler_stats_normalized.mean(axis=1)

    return batsman_stats_normalized, bowler_stats_normalized

def calculate_weighted_ratings(batsman_stats_normalized, bowler_stats_normalized):
    # Define the weights for the batters and bowlers
    batter_weights = {
        "total_runs": 0.3,
        "strike_rate": 0.3,
        "average": 0.2,
        "max_score": 0.2
    }

    bowler_weights = {
        "total_wickets": 0.3,
        "economy": 0.3,
        "average": 0.2,
        "max_wickets_in_match": 0.2
    }

    # Compute the weighted ratings
    batsman_stats_normalized["weighted_rating"] = batsman_stats_normalized[batter_weights.keys()].apply(
        lambda row: np.sum(row * list(batter_weights.values())), axis=1
    )
    bowler_stats_normalized["weighted_rating"] = bowler_stats_normalized[bowler_weights.keys()].apply(
        lambda row: np.sum(row * list(bowler_weights.values())), axis=1
    )

    return batsman_stats_normalized, bowler_stats_normalized

# Create a Kedro Data Catalog
data_catalog = DataCatalog(
    {
        "input_data": CSVDataSet(filepath="path/to/input_file.csv"),
        "batsman_stats": MemoryDataSet(),
        "bowler_stats": MemoryDataSet(),
        "batsman_stats_normalized": MemoryDataSet(),
        "bowler_stats_normalized": MemoryDataSet(),
        "batsman_strengths": MemoryDataSet(),
        "batsman_weaknesses": MemoryDataSet(),
        "bowler_strengths": MemoryDataSet(),
        "bowler_weaknesses": MemoryDataSet(),
    }
)
# Define Nodes
node_load_data = node(load_data,inputs=None, outputs="raw_data")
node_preprocess_data = node(preprocess_data, inputs="raw_data", outputs="preprocessed_data")
node_calculate_batsman_stats = node(calculate_batsman_stats, inputs="preprocessed_data", outputs="batsman_stats")
node_calculate_bowler_stats = node(calculate_bowler_stats, inputs="preprocessed_data", outputs="bowler_stats")
node_calculate_ratings = node(calculate_ratings, inputs=["batsman_stats", "bowler_stats"], outputs=["batsman_stats_normalized", "bowler_stats_normalized"])
node_calculate_weighted_ratings = node(calculate_weighted_ratings, inputs=["batsman_stats_normalized", "bowler_stats_normalized"], outputs=["batsman_stats_normalized_new", "bowler_stats_normalized_new"])

# Create a Kedro Pipeline
pipeline = Pipeline(
    [
        node_load_data,
        node_preprocess_data,
        node_calculate_batsman_stats,
        node_calculate_bowler_stats,
        node_calculate_ratings,
        # pip install kedro==0.16.5
        node_calculate_weighted_ratings,
    ],
    tags="pipeline_tag"  # Add any specific tag for the entire pipeline
)

# Run the Pipeline
runner = SequentialRunner()
result = runner.run(pipeline, data_catalog)


# Access the results
batsman_stats_normalized = result["batsman_stats_normalized_new"]
bowler_stats_normalized = result["bowler_stats_normalized_new"]

# (Optional) Save the results to CSV or any other format
batsman_stats_normalized.to_csv("batsman_stats_normalized_ratins.csv")
bowler_stats_normalized.to_csv("bowler_stats_normalized_ratings.csv")
