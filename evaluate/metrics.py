from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from icecream import ic
from utils.folder_utils import create_path

def calculate_correct_positive(actual, predicted):
    total_actual_positive = actual.count('positive')
    total_predicted_positive = predicted.count('positive')
    
    if total_actual_positive == 0:
        return 0.00
    
    correct_positive = sum(1 for a, p in zip(actual, predicted) if a == 'positive' and p == 'positive')
    percentage_correct_positive = (correct_positive / total_actual_positive) * 100
    
    return round(percentage_correct_positive, 2)

def calculate_correct_negative(actual, predicted):
    total_actual_negative = actual.count('negative')
    total_predicted_negative = predicted.count('negative')
    
    if total_actual_negative == 0:
        return 0.00
    
    correct_negative = sum(1 for a, p in zip(actual, predicted) if a == 'negative' and p == 'negative')
    percentage_correct_negative = (correct_negative / total_actual_negative) * 100
    
    return round(percentage_correct_negative, 2)

def calculate_positive_classified_as_negative(actual, predicted):
    total_actual_positive = actual.count('positive')
    
    if total_actual_positive == 0:
        return 0.00
    
    positive_classified_as_negative = sum(1 for a, p in zip(actual, predicted) if a == 'positive' and p == 'negative')
    percentage_positive_classified_as_negative = (positive_classified_as_negative / total_actual_positive) * 100
    
    return round(percentage_positive_classified_as_negative, 2)

def calculate_negative_classified_as_positive(actual, predicted):
    total_actual_negative = actual.count('negative')
    
    if total_actual_negative == 0:
        return 0.00
    
    negative_classified_as_positive = sum(1 for a, p in zip(actual, predicted) if a == 'negative' and p == 'positive')
    percentage_negative_classified_as_positive = (negative_classified_as_positive / total_actual_negative) * 100
    
    return round(percentage_negative_classified_as_positive, 2)


def evaluate_model(
        actual:list[str], 
        prediction:list[str], 
        save_file_path:str, 
        file_name:str
    )->None:
    """
    Evaluate sentiment analysis performance and write results to a text file.

    Parameters:
    actual (list): List of actual sentiment labels.
    prediction (list): List of predicted sentiment labels.
    save_file_path (str): Path to save the file.
    file_name (str): Name of the file.
    """

    # Calculating metrics
    precision = round(precision_score(actual, prediction, average='macro'), 2)
    recall = round(recall_score(actual, prediction, average='macro'), 2)
    accuracy = round(accuracy_score(actual, prediction), 2)
    f1 = round(f1_score(actual, prediction, average='macro'), 2)
    correct_positive_percentage = calculate_correct_positive(actual, prediction)
    correct_negative_percentage = calculate_correct_negative(actual, prediction)
    positive_classified_as_negative_percentage = calculate_positive_classified_as_negative(actual, prediction)
    negative_classified_as_positive_percentage = calculate_negative_classified_as_positive(actual, prediction)

    # Creating a DataFrame for the results
    results_df = pd.DataFrame({
        'Metric': [
            'Precision', 
            'Recall', 
            'Accuracy', 
            'F1 Score',
            'correct_positive_percentage',
            'correct_negative_percentage',
            'positive_classified_as_negative_percentage',
            'negative_classified_as_positive_percentage'
        ],
        'Value': [
            precision, 
            recall, 
            accuracy, 
            f1,
            correct_positive_percentage,
            correct_negative_percentage,
            positive_classified_as_negative_percentage,
            negative_classified_as_positive_percentage
        ]
    })

    ic(results_df)

    create_path(save_file_path)

    # Saving the results to a text file
    full_path = f"{save_file_path}/{file_name}.txt"
    with open(full_path, 'w') as file:
        file.write(results_df.to_string(index=False))

    return None

def plot_and_save_confusion_matrix(actual, predicted, save_file_path, file_name):
    """
    Calculate the confusion matrix, plot it using Matplotlib, and save the plot.

    Parameters:
    actual (list): List of actual labels.
    predicted (list): List of predicted labels.
    save_file_path (str): Path where the plot should be saved.
    file_name (str): Name of the file to save the plot.
    """

    # Calculate confusion matrix
    cm = confusion_matrix(actual, predicted, labels=["negative", "positive", "neutral"])

    # Plot using seaborn for a nicer looking heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=["negative", "positive", "neutral"], yticklabels=["negative", "positive", "neutral"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # Save the plot
    plt.savefig(f"{save_file_path}/{file_name}.png")

    # Close the plot
    plt.close()


