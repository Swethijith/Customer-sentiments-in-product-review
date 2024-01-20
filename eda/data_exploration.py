import pandas as pd
import matplotlib.pyplot as plt

def plot_review_sentiment_count_bars(
        value_counts:pd.Series,
        save_folder: str, 
        save_name:str,
        chart_title:str
    ):
    """
    Create a bar chart from value counts and save it to a given folder.

    Parameters:
        value_counts (pd.Series): Value counts data.
        save_folder (str): Folder path where the chart will be saved.
        chart_title (str, optional): Title for the bar chart.
    """
    # Create a DataFrame from the value counts series
    df = pd.DataFrame({'Values': value_counts.index, 'Counts': value_counts.values})
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df['Values'], df['Counts'])
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title(chart_title)
    plt.xticks(rotation=0)

    # Add count labels at the top of each bar
    for i, count in enumerate(df['Counts']):
        plt.text(df['Values'][i], count, str(count), ha='center', va='bottom', fontsize=12)
    
    # Save the chart to the specified folder
    save_path = f"{save_folder}/{save_name}.png"
    plt.savefig(save_path, bbox_inches='tight')

