import os
from icecream import ic
import pandas as pd

from evaluate.metrics import evaluate_model, plot_and_save_confusion_matrix
from results import results_dir_path

path = os.path.join(
    results_dir_path,
    'llm',
)
file_path = os.path.join(
    path,
    '_data.csv'
)
data = pd.read_csv(file_path)
data.dropna(inplace=True)
prediction = list(data.prediction)
actual = list(data.sentiment)


evaluate_model(
    actual=actual,
    prediction=prediction,
    save_file_path=path,
    file_name='evaluation_matrix.txt'
)

plot_and_save_confusion_matrix(
    actual=actual,
    predicted=prediction,
    save_file_path=path,
    file_name='confusion_matrix.png'
)