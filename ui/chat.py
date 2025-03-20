import streamlit as st
from core.retrieval import create_qa_chain, query_knowledge_base

def render_chat_ui(retriever):
    """
    Rendert die Chat-UI.
    
    Args:
        retriever: Der Retriever für die Abfrage
    """
    st.header("Chat")
    
    # Chat-Verlauf initialisieren, falls nicht vorhanden
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Bisherige Nachrichten anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
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
                    with st.expander("Quellen"):
                        for source in result['sources']:
                            st.markdown(f"- {source}")
                    
                    # Antwort zum Verlauf hinzufügen
                    st.session_state.messages.append({"role": "assistant", "content": result['answer']})