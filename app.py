import streamlit as st
from database.vectorstore import VectorStore
from ui.upload import render_upload_ui
from ui.chat import render_chat_ui

# Seitenkonfiguration
st.set_page_config(
    page_title="Wissens-Assistent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧠 Wissens-Assistent")
    
    # Initialisiere Vektorspeicher
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = VectorStore()
        st.session_state.data_sources = {
            "confluence": False,
            "pdf": False,
            "miro": False
        }
    
    # Render Upload-UI
    confluence_status, pdf_status, miro_status = render_upload_ui(st.session_state.vectorstore)
    
    # Status der Datenquellen aktualisieren
    if confluence_status:
        st.session_state.data_sources["confluence"] = True
    if pdf_status:
        st.session_state.data_sources["pdf"] = True
    if miro_status:
        st.session_state.data_sources["miro"] = True
    
    # Status der Wissensbasis anzeigen
    with st.sidebar:
        if st.session_state.vectorstore.vectorstore:
            st.success("Wissensbasis ist aktiv und bereit")
            
            # Zeige Datenquellen
            sources_text = []
            if st.session_state.data_sources["confluence"]:
                sources_text.append("Confluence")
            if st.session_state.data_sources["pdf"]:
                sources_text.append("PDF-Dokumente")
            if st.session_state.data_sources["miro"]:
                sources_text.append("Miro Boards")
            
            if sources_text:
                st.info(f"Aktive Datenquellen: {', '.join(sources_text)}")
    
    # Render Chat-UI
    render_chat_ui(st.session_state.vectorstore.get_retriever())

if __name__ == "__main__":
    main()