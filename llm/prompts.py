sentiment_analysis_prompt = """
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