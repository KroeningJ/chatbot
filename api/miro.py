import requests
import json
from langchain.schema import Document
from config.settings import MIRO_API_TOKEN

def fetch_miro_data(board_id):
    """
    Ruft Daten von einem Miro-Board ab.
    
    Args:
        board_id (str): Die ID des Miro-Boards
        
    Returns:
        dict: Die Daten des Miro-Boards oder None bei einem Fehler
    """
    url = f"https://api.miro.com/v2/boards/{board_id}/items"
    headers = {"Authorization": f"Bearer {MIRO_API_TOKEN}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Fehler beim Abrufen von Miro-Daten: {response.status_code} - {response.text}")

def extract_documents_from_miro(miro_data, board_id):
    """
    Extrahiert Dokumente mit Metadaten aus Miro-Daten.
    
    Args:
        miro_data (dict): Die Daten des Miro-Boards
        board_id (str): Die ID des Miro-Boards
        
    Returns:
        list: Liste von Document-Objekten mit Metadaten
    """
    documents = []
    
    for item in miro_data.get("data", []):
        item_type = item.get("type")
        item_id = item.get("id", "unknown_id")
        
        # Text aus verschiedenen möglichen Pfaden extrahieren
        text_content = None
        
        # Versuch 1: data.content
        if not text_content and "data" in item and "content" in item["data"]:
            text_content = item["data"]["content"]
        
        # Versuch 2: data.text
        if not text_content and "data" in item and "text" in item["data"]:
            text_content = item["data"]["text"]
            
        # Versuch 3: data.title
        if not text_content and "data" in item and "title" in item["data"]:
            text_content = item["data"]["title"]
        
        # Versuch 4: text direkt im Item
        if not text_content and "text" in item:
            text_content = item["text"]

        # Versuch 5: Plain Text in Shape-Item
        if not text_content and item_type == "shape" and "data" in item and "plainText" in item["data"]:
            text_content = item["data"]["plainText"]
            
        # Document-Objekt mit Metadaten erstellen
        if text_content:
            metadata = {
                "source": f"https://miro.com/app/board/{board_id}",
                "item_type": item_type,
                "item_id": item_id,
                "board_id": board_id
            }
            
            document = Document(
                page_content=text_content,
                metadata=metadata
            )
            documents.append(document)
    
    return documents

def extract_text_from_miro(miro_data):
    """
    Extrahiert Text aus Miro-Daten (Legacy-Funktion für Kompatibilität).
    
    Args:
        miro_data (dict): Die Daten des Miro-Boards
        
    Returns:
        list: Liste von extrahierten Texten
    """
    texts = []
    for item in miro_data.get("data", []):
        item_type = item.get("type")
        
        # Text aus verschiedenen möglichen Pfaden extrahieren
        text_content = None
        
        # Versuch 1: data.content
        if not text_content and "data" in item and "content" in item["data"]:
            text_content = item["data"]["content"]
        
        # Versuch 2: data.text
        if not text_content and "data" in item and "text" in item["data"]:
            text_content = item["data"]["text"]
            
        # Versuch 3: data.title
        if not text_content and "data" in item and "title" in item["data"]:
            text_content = item["data"]["title"]
        
        # Versuch 4: text direkt im Item
        if not text_content and "text" in item:
            text_content = item["text"]

        # Versuch 5: Plain Text in Shape-Item
        if not text_content and item_type == "shape" and "data" in item and "plainText" in item["data"]:
            text_content = item["data"]["plainText"]
            
        # Texte sammeln
        if text_content:
            texts.append(text_content)
    
    return texts