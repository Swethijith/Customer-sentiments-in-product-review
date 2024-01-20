import os
import json
import pandas as pd
from icecream import ic
from data import data_dir_path

from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataframe(df):
    """
    Split a dataframe into train, test, and validation sets.

    Parameters:
    df (pandas.DataFrame): The dataframe to be split.
    train_size (float): Proportion of the dataset to include in the train split.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the dataset to include in the validation split.

    Returns:
    tuple: Tuple containing three dataframes (train, test, validation).
    """
    train_size=0.7
    test_size=0.2
    val_size=0.1

    # First, split into train and temp (test + validation)
    train_df, temp_df = train_test_split(df, train_size=train_size)

    # Calculate the proportion of temp_df to be used for test to maintain overall test_size proportion
    proportion_of_temp_for_test = test_size / (test_size + val_size)

    # Split temp into test and validation
    test_df, val_df = train_test_split(temp_df, train_size=proportion_of_temp_for_test)

    return train_df, test_df, val_df

def balance_data(df:pd.DataFrame)->pd.DataFrame:

    # Min count of sentiment category
    min_count = df['sentiment'].value_counts().min()
    # min_count = 10000

    # Create a new DataFrame to store the balanced data
    balanced_df = pd.DataFrame()

    # Iterate through each category
    for category in df['sentiment'].unique():
        # Randomly sample 'min_count' reviews from each category
        sampled_reviews = df[df['sentiment'] == category].sample(min_count, random_state=42)
        
        # Append the sampled reviews to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, sampled_reviews], ignore_index=True)
    
    return balanced_df

def get_sentiment_from_rating(rating:float)->str:
    """Convert rating to sentiments"""

    rating = int(rating)
    if rating < 3:
        return "negative"
    elif rating > 3:
        return "positive"
    else:
        return "neutral"

# Path of json file of fashion product reviews
data_path = os.path.join(
    data_dir_path,
    "fashion_data\AMAZON_FASHION.json"
)

reviews = list()
with open(data_path,'r') as file:
    for row in file:
        reviews.append(json.loads(row))
    review_data = pd.DataFrame(reviews)[['overall','reviewText']]

ic(review_data.isna().sum())

# Drop null values
review_data.dropna(inplace=True)

# Reset index
review_data.reset_index(inplace=True, drop=True)

review_data['sentiment'] = review_data['overall'].apply(get_sentiment_from_rating)
ic(len(review_data), "reviews loaded.")

balanced_review_data = balance_data(review_data)
ic(len(balanced_review_data), "reviews available.")

train_data, val_data, test_data = split_dataframe(balanced_review_data)

