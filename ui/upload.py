import streamlit as st
from api.confluence import load_confluence_documents
from api.pdf import load_pdf_document, save_uploaded_pdf
from api.miro import fetch_miro_data, extract_text_from_miro

def render_upload_ui(vectorstore):
    """
    Rendert die Upload-UI.
    
    Args:
        vectorstore: Der Vektorspeicher
        
    Returns:
        tuple: (bool, bool, bool) Statuswerte für Confluence, PDF und Miro
    """
    st.sidebar.header("Dokumente hinzufügen")
    
    # Status-Variablen
    confluence_status = False
    pdf_status = False
    miro_status = False
    
    # Confluence-Bereich
    st.sidebar.subheader("Confluence")
    load_confluence = st.sidebar.toggle("Confluence Dokumente laden", value=True)
    
    # PDF-Upload-Bereich
    st.sidebar.subheader("PDF Dokumente")
    pdf_files = st.sidebar.file_uploader(
        "PDFs hochladen", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Miro-Integration
    st.sidebar.subheader("Miro Boards")
    miro_board_id = st.sidebar.text_input("Miro Board ID")
    
    # Verarbeiten-Button
    process_docs = st.sidebar.button("Dokumente verarbeiten")
    
    if process_docs:
        success = False
        
        # Confluence Dokumente laden
        if load_confluence:
            with st.spinner("Lade Confluence-Dokumente..."):
                try:
                    confluence_docs = load_confluence_documents()
                    if confluence_docs:
                        if vectorstore.add_documents(confluence_docs):
                            st.success("Confluence-Dokumente erfolgreich verarbeitet!")
                            confluence_status = True
                            success = True
                except Exception as e:
                    st.error(f"Fehler beim Laden von Confluence-Dokumenten: {e}")
        
        # PDF-Dokumente laden
        if pdf_files:
            with st.spinner("Lade PDF-Dokumente..."):
                pdf_docs = []
                for pdf_file in pdf_files:
                    try:
                        # PDF speichern und laden
                        temp_path = save_uploaded_pdf(pdf_file)
                        pdf_docs.extend(load_pdf_document(temp_path))
                    except Exception as e:
                        st.error(f"Fehler beim Laden von {pdf_file.name}: {e}")
                
                # Dokumente zum Vektorspeicher hinzufügen
                if pdf_docs:
                    if vectorstore.add_documents(pdf_docs):
                        st.success("PDF-Dokumente erfolgreich verarbeitet!")
                        pdf_status = True
                        success = True
        
        # Miro-Daten laden
        if miro_board_id:
            with st.spinner("Lade Miro-Daten..."):
                try:
                    miro_data = fetch_miro_data(miro_board_id)
                    if miro_data:
                        texts = extract_text_from_miro(miro_data)
                        if texts and vectorstore.add_texts(texts):
                            st.success("Miro-Daten erfolgreich verarbeitet!")
                            miro_status = True
                            success = True
                except Exception as e:
                    st.error(f"Fehler beim Laden von Miro-Daten: {e}")
        
        if not success:
            st.warning("Keine Dokumente zum Verarbeiten ausgewählt oder alle Verarbeitungen waren erfolglos.")
    
    return confluence_status, pdf_status, miro_status