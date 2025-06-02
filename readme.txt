// Für später

# Wissens-Assistent

Ein interaktiver Assistent zur Indexierung und Abfrage von Wissen aus verschiedenen Quellen wie Confluence, PDF-Dokumenten und Miro-Boards.

## Projektstruktur

```
📂 projekt-root/
│── 📂 src/                # Hauptcode des Projekts
│   │── 📂 api/            # API-Handler für verschiedene Datenquellen
│   │   │── confluence.py  # Funktionen zur Confluence-Datenextraktion
│   │   │── miro.py        # Funktionen zur Miro-Datenextraktion
│   │   │── pdf.py         # Funktionen zur Verarbeitung von PDFs
│   │── 📂 database/       # Datenbankschnittstellen
│   │   │── vectorstore.py # ChromaDB-Verwaltung
│   │── 📂 ui/             # Streamlit UI-Module
│   │   │── chat.py        # Chatbot-Interface
│   │   │── upload.py      # Datei-Upload-UI
│   │── 📂 core/           # Kernfunktionen & Logik
│   │   │── embeddings.py  # OpenAI Embeddings-Verwaltung
│   │   │── retrieval.py   # Retrieval-Mechanismus für Abfragen
│── 📂 config/             # Konfigurationsdateien
│   │── settings.py        # API-Keys & allgemeine Konfigs
│── app.py                 # Haupt-Streamlit-Anwendung
│── requirements.txt       # Abhängigkeiten
│── README.md              # Projektdokumentation
```

## Installation

1. Repository klonen:
```
git clone <repository-url>
cd wissens-assistent
```

2. Abhängigkeiten installieren:
```
pip install -r requirements.txt
```

3. API-Keys konfigurieren:
Bearbeite die Datei `config/settings.py` und setze die erforderlichen API-Keys.

## Verwendung

1. Starte die Anwendung:
```
streamlit run app.py aktuell nur über venv mit folgendem Befehl: venv\Scripts\activate

```

2. Öffne die Anwendung im Browser (normalerweise unter http://localhost:8501).

3. Wähle die gewünschten Datenquellen aus und klicke auf "Dokumente verarbeiten".

4. Stelle Fragen an die Wissensbasis über das Chat-Interface.

## Funktionen

- Integration mit Confluence zur Extraktion von Wiki-Inhalten
- Verarbeitung von PDF-Dokumenten
- Integration mit Miro-Boards zur Extraktion von Inhalten
- Vektorbasierte Suche mit ChromaDB
- Conversational AI mit OpenAI

## Fehlerbehebung

- Bei Problemen mit der Verbindung zu externen APIs überprüfe die Konfiguration in `config/settings.py`.
- Stelle sicher, dass die erforderlichen API-Keys gültig sind.
- Bei Fehlern in der Anwendung werden diese in der Streamlit-Oberfläche angezeigt.