from langchain_openai import OpenAIEmbeddings
from config.settings import OPENAI_API_KEY

def get_embedding_function():
    """
    Erstellt eine Embedding-Funktion mit OpenAI.
    
    Returns:
        OpenAIEmbeddings: Die Embedding-Funktion
    """
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)