from langchain.prompts import PromptTemplate
from langchain.llms import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from icecream import ic
from langchain.llms import OpenAI
import json
from pydantic import BaseModel, validator

from llm.prompts import sentiment_analysis_prompt

class SentimentModel(BaseModel):
    sentiment: str

    @validator('sentiment')
    def match_sentiment(cls, v):
        allowed_values = ["positive", "negative", "neutral"]
        matched_value = next((val for val in allowed_values if val in v.lower()), None)
        return matched_value

# Example Usage
model = SentimentModel(sentiment="I feel very positive about this!")
print(model.sentiment)  # Output will be "positive" if it's found in the input string


load_dotenv()

llm = GooglePalm(temperature=0.0)
chat_llm = ChatGoogleGenerativeAI(model="gemini-pro")

# gpt_llm = OpenAI(model_name="text-davinci-003", temperature=0.0)

def predict_sentiment(review:str):

    try:

        prompt = PromptTemplate(
            template=sentiment_analysis_prompt,
            input_variables=["review"],
        )

        chain = prompt | llm 
        result = chain.invoke({
                "review" : review,
        })

    except Exception as e:
        ic("In exception")
        prompt = """
            Analyze the sentiment of the following customer review. 
            Note that the review may contain words that typically have a sensitive connotation, 
            but here they are used in the context of describing clothing or fashion items. 
            Your task is to interpret these words correctly within this context and 
            determine the overall sentiment of the review - 
            whether it is positive, negative, or neutral. 
            Please provide a clear sentiment label (positive/negative/neutral), focusing solely on the customer's 
            satisfaction or dissatisfaction with the clothing item. Provide the sentiment label only.

            Review: {review}
        """
        output = chat_llm.invoke(prompt).content.lower()
        result = SentimentModel(sentiment=output).sentiment
        ic(output)
        ic(result)

    return result

if __name__ == "__main__":
    from icecream import ic

    review = """
        'Size ordered fits as expected. Looked real nice when received.  After a week '
             'of wearing it its pretty scratched up. Scratches real easy. Very light an '
             'almost plastic feeling. But hey its a 14 dollar ring. Over all i like it' 
    """
    prompt = f"""
        Please analyze the sentiment of the given fashion product review and 
        classify it as either positive, negative, or neutral. 
        Please provide a detailed response that accurately represents the user's sentiment. 
        Provide the answer in a single word.

        Product Review: {review}

        Sentiment:

    """
    ic(google_llm.invoke(prompt))

    prompt = PromptTemplate(
        template=sentiment_analysis_prompt,
        input_variables=["review"],
    )

    chain = prompt | google_llm 
    result = chain.invoke({
        "review" : review,
    })
    ic(result)