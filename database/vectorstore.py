import os
import sys
sys.path.append(r'c:\Users\jnskr\Desktop\Chatbot')
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import OPENAI_API_KEY, CHROMADB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

class VectorStore:
    def __init__(self):
        """Initialisiert den Vektorspeicher mit OpenAI Embeddings"""
        self.embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.try_load_existing_vectorstore()
    
    def try_load_existing_vectorstore(self):
        """Versucht, einen existierenden Vektorspeicher zu laden, falls vorhanden"""
        try:
            if os.path.exists(CHROMADB_DIR):
                self.vectorstore = Chroma(
                    persist_directory=CHROMADB_DIR,
                    embedding_function=self.embedding_function
                )
                return True
        except Exception as e:
            raise Exception(f"Fehler beim Laden des existierenden Vektorspeichers: {e}")
        return False
    
    def get_retriever(self):
        """Gibt den Retriever für die Abfrage zurück"""
        if self.vectorstore:
            return self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        return None
    
    def add_documents(self, documents):
        """
        Fügt Dokumente zum Vektorspeicher hinzu.
        
        Args:
            documents (list): Liste von Dokumenten
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not documents:
            return False
        
        try:
            # Teile Dokumente in Chunks
            docs = self.text_splitter.split_documents(documents)
            
            if self.vectorstore:
                # Füge Dokumente zum bestehenden Vektorspeicher hinzu
                self.vectorstore.add_documents(docs)
            else:
                # Erstelle neuen Vektorspeicher
                self.vectorstore = Chroma.from_documents(
                    docs,
                    self.embedding_function,
                    persist_directory=CHROMADB_DIR
                )
            
            return True
        except Exception as e:
            raise Exception(f"Fehler beim Hinzufügen von Dokumenten zum Vektorspeicher: {e}")
    
    def add_texts(self, texts):
        """
        Fügt Texte zum Vektorspeicher hinzu.
        
        Args:
            texts (list): Liste von Texten
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        if not texts:
            return False
        
        try:
            # Erstelle Dokumente aus Texten
            docs = self.text_splitter.create_documents(texts)
            
            if self.vectorstore:
                # Füge Dokumente zum bestehenden Vektorspeicher hinzu
                self.vectorstore.add_documents(docs)
            else:
                # Erstelle neuen Vektorspeicher
                self.vectorstore = Chroma.from_documents(
                    docs,
                    self.embedding_function,
                    persist_directory=CHROMADB_DIR
                )
            
            return True
        except Exception as e:
            raise Exception(f"Fehler beim Hinzufügen von Texten zum Vektorspeicher: {e}")