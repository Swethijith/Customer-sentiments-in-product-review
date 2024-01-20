import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from icecream import ic
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from data.dataloader import test_data
from evaluate.metrics import evaluate_model, plot_and_save_confusion_matrix
from results import results_dir_path

tqdm.pandas()

# Download stopwords
nltk.download('stopwords')

# Download WordNet lemmatizer data
nltk.download('wordnet')

# Download Punkt tokenizer models
nltk.download('punkt')

# Download the VADER lexicon
nltk.download('vader_lexicon')

def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(review_text:str):

    score = sia.polarity_scores(review_text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

test_data['prediction'] = test_data['reviewText'].progress_apply(analyze_sentiment)

prediction = list(test_data['prediction'])
actual = list(test_data['sentiment'])
save_file_path = os.path.join(
    results_dir_path,
    "vader"
)

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

test_data['processedReviewText'] = test_data['reviewText'].progress_apply(preprocess_text)
test_data['predictionProcessed'] = test_data['processedReviewText'].progress_apply(analyze_sentiment)

prediction = list(test_data['predictionProcessed'])
save_file_path = os.path.join(
    results_dir_path,
    "vader"
)

evaluate_model(
    actual=actual,
    prediction=prediction,
    save_file_path=save_file_path,
    file_name="evaluation_matrix_processed"
)

plot_and_save_confusion_matrix(
    actual=actual,
    predicted=prediction,
    save_file_path=save_file_path,
    file_name="confusion_matrix_processed"
)

csv_file_path = os.path.join(save_file_path,"data.csv")
test_data.to_csv(csv_file_path, index=False)