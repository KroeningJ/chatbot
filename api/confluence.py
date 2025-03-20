from langchain_community.document_loaders import ConfluenceLoader
from config.settings import (
    CONFLUENCE_API_KEY, 
    CONFLUENCE_URL, 
    CONFLUENCE_USERNAME, 
    CONFLUENCE_SPACE_KEY,
    CONFLUENCE_LIMIT
)

def load_confluence_documents():
    """
    Lädt Dokumente aus Confluence.
    
    Returns:
        list: Liste von Dokumenten aus Confluence
    """
    try:
        loader = ConfluenceLoader(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY,
            include_attachments=True,
            limit=CONFLUENCE_LIMIT
        )
        return loader.load()
    except Exception as e:
        raise Exception(f"Fehler beim Laden von Confluence-Dokumenten: {e}")