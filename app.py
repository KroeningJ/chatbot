import streamlit as st
from database.vectorstore import VectorStore
from database.datastore import ChatDataStore
from ui.upload import render_upload_ui
from ui.chat import render_chat_ui

# Seitenkonfiguration
st.set_page_config(
    page_title="Wissens-Assistent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_session_management():
    """Rendert die Session-Management Sektion in der Sidebar"""
    with st.sidebar.expander("💬 Chat-Sessions"):
        st.markdown("### Session-Verwaltung")
        
        # Neue Session Button
        if st.button("🔄 Neue Chat-Session starten"):
            # Reset Chat-Verlauf
            st.session_state.messages = []
            # Neue Session-ID generieren
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
            # Neue Session in Datenbank erstellen
            st.session_state.datastore.create_session(
                st.session_state.session_id,
                st.session_state.data_sources
            )
            st.rerun()
        
        # Alle Sessions anzeigen
        if st.button("📋 Alle Sessions anzeigen"):
            try:
                sessions = st.session_state.datastore.get_all_sessions()
                
                if sessions:
                    st.markdown("#### Letzte 10 Sessions:")
                    for session in sessions[:10]:
                        st.text(f"Session: {session['session_id'][:8]}...")
                        st.text(f"Nachrichten: {session['message_count']}")
                        st.text(f"Letzte Aktivität: {session['last_updated']}")
                        st.divider()
                else:
                    st.info("Keine Chat-Sessions vorhanden")
                    
            except Exception as e:
                st.error(f"Fehler beim Abrufen der Sessions: {e}")

def main():
    st.title("🧠 Wissens-Assistent")
    
    # Initialisiere Vektorspeicher und Datastore
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = VectorStore()
        st.session_state.datastore = ChatDataStore()
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
        
        # Session-Management Sektion
        render_session_management()
    
    # Render Chat-UI
    render_chat_ui(st.session_state.vectorstore.get_retriever())

if __name__ == "__main__":
    main()