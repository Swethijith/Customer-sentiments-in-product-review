from langchain.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv

load_dotenv()

def embedding_function():
    
    # Initialize GooglePalmEmbeddings
    embeddings = GooglePalmEmbeddings()

    return embeddings