import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(file_path):
    """
    Lädt ein PDF-Dokument.
    
    Args:
        file_path (str): Der Pfad zur PDF-Datei
        
    Returns:
        list: Liste von Dokumenten aus der PDF
    """
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        raise Exception(f"Fehler beim Laden der PDF-Datei {os.path.basename(file_path)}: {e}")

def save_uploaded_pdf(uploaded_file):
    """
    Speichert eine hochgeladene PDF-Datei temporär.
    
    Args:
        uploaded_file: Das hochgeladene Datei-Objekt
        
    Returns:
        str: Der Pfad zur gespeicherten Datei
    """
    try:
        # Erstelle temp Ordner, falls nicht vorhanden
        os.makedirs("temp", exist_ok=True)
        
        # Temporäre Datei speichern
        temp_path = os.path.join("temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        raise Exception(f"Fehler beim Speichern der PDF-Datei: {e}")