import streamlit as st
from core.retrieval import create_qa_chain, query_knowledge_base
from database.datastore import ChatDataStore
import uuid

def render_chat_ui(retriever):
    """
    Rendert die Chat-UI.
    
    Args:
        retriever: Der Retriever für die Abfrage
    """
    st.header("Chat")
    
    # Chat-Verlauf und Session initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # Initialisiere ChatDataStore und erstelle Session
        if 'datastore' not in st.session_state:
            st.session_state.datastore = ChatDataStore()
        st.session_state.datastore.create_session(
            st.session_state.session_id,
            st.session_state.data_sources
        )
    
    # Bisherige Nachrichten anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Zeige Quellen für Assistant-Nachrichten
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Quellen"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Eingabefeld für Nutzeranfrage
    user_query = st.chat_input("Stelle eine Frage...")
    
    if user_query:
        # Überprüfe, ob Retriever bereit ist
        if not retriever:
            st.warning("Bitte laden und verarbeiten Sie zuerst Dokumente, bevor Sie Fragen stellen können.")
            return
        
        # Nutzernachricht anzeigen
        with st.chat_message("user"):
            st.write(user_query)
        
        # Nachricht zum Verlauf hinzufügen
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Nachricht in Datenbank speichern
        st.session_state.datastore.add_message(
            st.session_state.session_id,
            "user",
            user_query
        )
        
        # Antwort generieren
        with st.chat_message("assistant"):
            with st.spinner("Generiere Antwort..."):
                # QA Chain erstellen
                qa_chain = create_qa_chain(retriever)
                
                # Abfrage durchführen
                result = query_knowledge_base(qa_chain, user_query)
                
                if result:
                    st.write(result['answer'])
                    
                    # Quellenangaben
                    if result['sources']:
                        with st.expander("Quellen"):
                            for source in result['sources']:
                                st.markdown(f"- {source}")
                    
                    # Antwort zum Verlauf hinzufügen
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                    # Antwort in Datenbank speichern
                    st.session_state.datastore.add_message(
                        st.session_state.session_id,
                        "assistant",
                        result['answer'],
                        result['sources']
                    )