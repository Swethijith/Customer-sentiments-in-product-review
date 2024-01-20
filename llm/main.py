from tqdm import tqdm
import os

from data.dataloader import test_data
from evaluate.metrics import evaluate_model, plot_and_save_confusion_matrix
from results import results_dir_path
from llm.prompt_analyser import predict_sentiment

tqdm.pandas()

test_data['prediction'] = test_data['reviewText'].progress_apply(predict_sentiment)


prediction = list(test_data['prediction'])
actual = list(test_data['sentiment'])
save_file_path = os.path.join(
    results_dir_path,
    "llm"
)

csv_file_path = os.path.join(save_file_path,"_data.csv")
test_data.to_csv(csv_file_path, index=False)

evaluate_model(
    actual=actual,
    prediction=prediction,
    save_file_path=save_file_path,
    file_name="evaluation_matrix"
)

plot_and_save_confusion_matrix(
    actual=actual,
    predicted=prediction,
    save_file_path=save_file_path,
    file_name="confusion_matrix"
)

csv_file_path = os.path.join(save_file_path,"data.csv")
test_data.to_csv(csv_file_path, index=False)