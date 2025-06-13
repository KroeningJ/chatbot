from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from config.settings import OPENAI_API_KEY, LLM_TEMPERATURE

def create_qa_chain(retriever):
    """
    Erstellt eine RetrievalQA Chain.
    
    Args:
        retriever: Der Retriever für die Abfrage
        
    Returns:
        RetrievalQA: Die RetrievalQA Chain
    """
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=LLM_TEMPERATURE)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def query_knowledge_base(qa_chain, query):
    """
    Führt eine Abfrage auf der Wissensbasis durch.
    
    Args:
        qa_chain: Die RetrievalQA Chain
        query (str): Die Abfrage
        
    Returns:
        dict: Das Ergebnis der Abfrage mit Antwort und Quellen
    """
    try:
        result = qa_chain(query)
        
        # Quellen deduplizieren - nur einzigartige Quellen anzeigen
        sources = []
        seen_sources = set()
        
        for doc in result.get('source_documents', []):
            source = doc.metadata.get('source', 'Unbekannt')
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)
        
        return {
            "answer": result['result'],
            "sources": sources
        }
    except Exception as e:
        raise Exception(f"Fehler bei der Abfrage: {e}")