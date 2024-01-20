import streamlit as st
from llm.prompt_analyser import predict_sentiment

# Streamlit UI elements
st.title("Product Review Sentiment Analysis")
review_text = st.text_area("Enter your product review here:",  height=50)
predict_button = st.button("Predict Sentiment")

# Perform sentiment analysis when the button is clicked
if predict_button:
    if review_text.strip() == "":
        st.error("Please enter a review before predicting sentiment.")
    else:
        sentiment = predict_sentiment(review_text)
        st.header(f"Sentiment: {sentiment}")
