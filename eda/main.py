import os
from icecream import ic
from eda.data_exploration import plot_review_sentiment_count_bars
from data.dataloader import review_data, balanced_review_data
from results import results_dir_path
from utils.folder_utils import create_path


# Review count bar plot
save_folder = os.path.join(results_dir_path,"eda")
create_path(save_folder)
save_name = "sentiments_counts"
plot_review_sentiment_count_bars(
    value_counts=review_data.sentiment.value_counts(),
    save_folder=save_folder,
    save_name=save_name,
    chart_title="Sentiments count bar plot"
)

# Balanced review count bar plot
save_folder = os.path.join(results_dir_path,"eda")
create_path(save_folder)
save_name = "balanced_sentiments_counts"
plot_review_sentiment_count_bars(
    value_counts=balanced_review_data.sentiment.value_counts(),
    save_folder=save_folder,
    save_name=save_name,
    chart_title="Balanced data sentiments count bar plot"
)