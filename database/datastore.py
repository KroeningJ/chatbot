import sqlite3
import json
from datetime import datetime
import os
from config.settings import CHROMADB_DIR

class ChatDataStore:
    def __init__(self):
        """Initialisiert die SQLite-Datenbank für Chat-Verläufe"""
        # Erstelle sql_db Verzeichnis
        db_dir = "sql_db"
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "chat_history.db")
        self._init_database()
    
    def _init_database(self):
        """Erstellt die Datenbanktabellen, falls sie nicht existieren"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabelle für Chat-Sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active_sources TEXT
                )
            ''')
            
            # Tabelle für einzelne Nachrichten
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                )
            ''')
            
            # Tabelle für Bewertungen (für spätere Evaluation)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    rating INTEGER,
                    feedback TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES chat_messages(message_id)
                )
            ''')
            
            conn.commit()
    
    def create_session(self, session_id, active_sources=None):
        """
        Erstellt eine neue Chat-Session.
        
        Args:
            session_id (str): Eindeutige Session-ID
            active_sources (dict): Aktive Datenquellen
            
        Returns:
            bool: True bei Erfolg
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_sessions (session_id, active_sources)
                    VALUES (?, ?)
                ''', (session_id, json.dumps(active_sources or {})))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Session existiert bereits
            return True
        except Exception as e:
            raise Exception(f"Fehler beim Erstellen der Session: {e}")
    
    def add_message(self, session_id, role, content, sources=None):
        """
        Fügt eine Nachricht zum Chat-Verlauf hinzu.
        
        Args:
            session_id (str): Session-ID
            role (str): Rolle (user/assistant)
            content (str): Nachrichteninhalt
            sources (list): Liste der verwendeten Quellen
            
        Returns:
            int: Message-ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Nachricht hinzufügen
                cursor.execute('''
                    INSERT INTO chat_messages (session_id, role, content, sources)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, role, content, json.dumps(sources or [])))
                
                # Session aktualisieren
                cursor.execute('''
                    UPDATE chat_sessions 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                ''', (session_id,))
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            raise Exception(f"Fehler beim Speichern der Nachricht: {e}")
    
    def get_session_history(self, session_id):
        """
        Holt den kompletten Chat-Verlauf einer Session.
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            list: Liste von Nachrichten
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_id, role, content, sources, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                ''', (session_id,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'message_id': row[0],
                        'role': row[1],
                        'content': row[2],
                        'sources': json.loads(row[3]) if row[3] else [],
                        'timestamp': row[4]
                    })
                
                return messages
        except Exception as e:
            raise Exception(f"Fehler beim Abrufen des Chat-Verlaufs: {e}")
    
    def get_all_sessions(self):
        """
        Holt alle Chat-Sessions.
        
        Returns:
            list: Liste aller Sessions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id, created_at, last_updated, active_sources,
                           (SELECT COUNT(*) FROM chat_messages WHERE session_id = cs.session_id) as message_count
                    FROM chat_sessions cs
                    ORDER BY last_updated DESC
                ''')
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row[0],
                        'created_at': row[1],
                        'last_updated': row[2],
                        'active_sources': json.loads(row[3]) if row[3] else {},
                        'message_count': row[4]
                    })
                
                return sessions
        except Exception as e:
            raise Exception(f"Fehler beim Abrufen der Sessions: {e}")
    
    def add_rating(self, message_id, rating, feedback=None):
        """
        Fügt eine Bewertung für eine Nachricht hinzu.
        
        Args:
            message_id (int): Nachrichten-ID
            rating (int): Bewertung (z.B. 1-5)
            feedback (str): Optionales Feedback
            
        Returns:
            bool: True bei Erfolg
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO message_ratings (message_id, rating, feedback)
                    VALUES (?, ?, ?)
                ''', (message_id, rating, feedback))
                conn.commit()
                return True
        except Exception as e:
            raise Exception(f"Fehler beim Speichern der Bewertung: {e}")
    
    def get_messages_for_evaluation(self, limit=None):
        """
        Holt Nachrichten für die Evaluation.
        
        Args:
            limit (int): Maximale Anzahl von Nachrichten
            
        Returns:
            list: Liste von Q&A-Paaren mit Quellen
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT m1.content as question, m2.content as answer, m2.sources
                    FROM chat_messages m1
                    JOIN chat_messages m2 ON m1.session_id = m2.session_id 
                        AND m2.message_id = (
                            SELECT MIN(message_id) 
                            FROM chat_messages 
                            WHERE session_id = m1.session_id 
                            AND message_id > m1.message_id 
                            AND role = 'assistant'
                        )
                    WHERE m1.role = 'user'
                '''
                
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor.execute(query)
                
                qa_pairs = []
                for row in cursor.fetchall():
                    qa_pairs.append({
                        'question': row[0],
                        'answer': row[1],
                        'sources': json.loads(row[2]) if row[2] else []
                    })
                
                return qa_pairs
        except Exception as e:
            raise Exception(f"Fehler beim Abrufen der Evaluationsdaten: {e}")