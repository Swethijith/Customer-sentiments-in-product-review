from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from tqdm import tqdm
from icecream import ic

from data.dataloader import (
    get_sentiment_from_rating, 
    test_data
)
from evaluate.metrics import (
    plot_and_save_confusion_matrix,
    evaluate_model
)
from results import results_dir_path

tqdm.pandas()

tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

def predict_sentiment(review:str)->str:

    try:
        inputs = tokenizer(review, return_tensors="pt",truncation=True)

        # Make prediction
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_index = torch.argmax(logits, dim=1)
        sentiment = get_sentiment_from_rating(prediction_index.item())

        return sentiment

    except Exception as e:
        ic(review)
        ic(e)

        return "neutral"

save_file_path = os.path.join(
    results_dir_path,
    "bert"
)

test_data['prediction'] = test_data['reviewText'].progress_apply(predict_sentiment)
csv_file_path = os.path.join(save_file_path,"_data.csv")
test_data.to_csv(csv_file_path, index=False)

prediction = list(test_data['prediction'])
actual = list(test_data['sentiment'])

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